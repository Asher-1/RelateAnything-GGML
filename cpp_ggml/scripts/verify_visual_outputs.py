#!/usr/bin/env python3
"""Check offline charts, all inference selections and the extracted test ZIP.

Requires Playwright and an installed Chrome for validation only. Inference,
packaging and viewing the resulting HTML have no browser automation dependency.
"""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile

from package_test_data import PACKAGE, DATA, digest_file, verify


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--chrome', default='/opt/google/chrome/chrome')
    parser.add_argument('--gallery', type=Path, default=DATA.parent / 'benchmarks/gallery')
    parser.add_argument('--archive', type=Path,
                        default=DATA.parent / 'outputs/packages' / (PACKAGE + '.zip'))
    args = parser.parse_args()
    gallery, archive = args.gallery.resolve(), args.archive.resolve()
    descriptor = json.loads(archive.with_suffix('.package.json').read_text())
    verify(archive, descriptor)
    errors, failures, checks, mobile = [], [], [], []
    with tempfile.TemporaryDirectory(prefix='ra-test-preview-') as temporary, sync_playwright() as automation:
        with zipfile.ZipFile(archive) as packed:
            packed.extractall(temporary)  # All member paths/modes/hashes verified above.
        extracted = Path(temporary) / PACKAGE
        browser = automation.chromium.launch(executable_path=args.chrome, headless=True,
                                             args=['--disable-gpu'])
        version = browser.version
        page = browser.new_page(viewport={'width': 1580, 'height': 1100})
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('requestfailed', lambda request: failures.append(
            {'url': request.url, 'error': request.failure}))
        page.goto((gallery / 'index.html').as_uri(), wait_until='load')
        models = page.locator('#model option').evaluate_all('(v)=>v.map(x=>x.value)')
        assert len(models) == 3
        for model in models:
            page.select_option('#model', model)
            configs = page.locator('#config option').evaluate_all('(v)=>v.map(x=>x.value)')
            assert len(configs) == 10 and page.locator('#sample option').count() == 50
            for config in configs:
                page.select_option('#config', config)
                for image_id in range(50):
                    page.select_option('#sample', str(image_id))
                    assert page.locator('#panels svg').count() == 3
                    assert page.locator('#relations tr').count() > 0
                    assert model + ' / ' + config in page.locator('#stats').inner_text()
                page.locator('#numeric').evaluate('(e)=>e.decode()')
                page.locator('#distribution').evaluate('(e)=>e.decode()')
                checks.append({'model': model, 'config': config, 'samples': 50})
        for figure in gallery.glob('*.png'):
            size = page.evaluate('async u=>{const i=new Image();i.src=u;await i.decode();'
                                 'return [i.naturalWidth,i.naturalHeight]}', figure.as_uri())
            assert size[0] >= 1000 and size[1] >= 600, figure
        page.screenshot(path=str(archive.parent / 'inference_preview.png'))
        for width in (390, 768):
            page.set_viewport_size({'width': width, 'height': 1000})
            actual = page.evaluate('document.documentElement.scrollWidth')
            assert actual <= width, ('gallery overflow', width, actual)
            mobile.append({'width': width, 'document_width': actual})
        page.set_viewport_size({'width': 1400, 'height': 1050})
        page.goto((extracted / 'index.html').as_uri(), wait_until='load')
        catalog = json.loads((extracted / 'catalog.json').read_text())
        for index, sample in enumerate(catalog['samples']):
            page.select_option('#suite', sample['suite'])
            page.select_option('#sample', str(index))
            picture = page.locator(f'#picture svg[data-sample="{index}"]')
            picture.wait_for(state='visible')
            assert picture.locator('rect').count() == len(sample['objects'])
            fields = json.loads(page.locator('#autofill').inner_text())
            assert fields == {name: sample[name] for name in ('image', 'boxes_xyxy', 'object_names')}
            assert page.locator('#original').get_attribute('href') == sample['image']
        page.select_option('#suite', 'ra4m_50')
        page.select_option('#sample', '0')
        page.locator('#picture svg[data-sample="0"]').wait_for(state='visible')
        page.screenshot(path=str(archive.parent / 'test_data_preview.png'))
        for width in (390, 768):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth') <= width
        page.goto((DATA.parent / 'outputs/index.html').as_uri(), wait_until='load')
        for link in page.locator('a').evaluate_all('(v)=>v.map(x=>x.href)'):
            if link.startswith('file:'):
                from urllib.parse import unquote, urlsplit
                assert Path(unquote(urlsplit(link).path)).is_file(), link
        for image in page.locator('img').all():
            image.evaluate('(e)=>e.decode()')
        comparison = DATA.parent/'benchmarks/ALL_COMPARISONS.html'
        page.goto(comparison.as_uri(), wait_until='load')
        pictures = page.locator('img').evaluate_all('(v)=>v.map(x=>x.src)')
        optimization = DATA.parent/'benchmarks/optimization/summary.json'
        extra = len(json.loads(optimization.read_text())['figures']) if optimization.is_file() else 0
        assert len(pictures) == 192+extra and page.locator('details').count() == 150
        for picture in pictures:
            size = page.evaluate('async u=>{const i=new Image();i.src=u;await i.decode();'
                                 'return [i.naturalWidth,i.naturalHeight]}', picture)
            assert min(size) > 0, picture
        for width in (390, 768):
            page.set_viewport_size({'width': width, 'height': 1000})
            assert page.evaluate('document.documentElement.scrollWidth') <= width
        browser.close()
    actual_failures = [row for row in failures if row['error'] != 'net::ERR_ABORTED']
    assert not errors, errors
    assert not actual_failures, actual_failures
    audit = {'status': 'pass', 'browser_version': version, 'checks': checks,
             'rendered_model_config_samples': sum(row['samples'] for row in checks),
             'png_figures': len(list(gallery.glob('*.png'))),
             'svg_figures': len(list(gallery.glob('*.svg'))), 'mobile_checks': mobile,
             'index_sha256': digest_file(gallery / 'index.html'),
             'test_zip_sha256': descriptor['zip_sha256'], 'test_samples': descriptor['samples'],
             'inference_ready_samples': descriptor['inference_ready_samples'],
             'comparison_index_sha256': digest_file(comparison),
             'comparison_index_images': len(pictures), 'original_image_grids': 150,
             'page_errors': errors, 'failed_requests': actual_failures}
    (gallery / 'browser_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    archive.with_suffix('.browser_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({key: value for key, value in audit.items() if key != 'checks'}))


if __name__ == '__main__':
    main()
