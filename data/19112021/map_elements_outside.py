import arff, sys

from PIL import Image

outside = []
removed_website = 0
new_data = []

with open('./data/dataset.unclassified.arff') as f:
    dataset = arff.load(f)
    attributes = [ attr[0] for attr in dataset['attributes'] ]

    for row in dataset['data']:
        base_screenshot = row[attributes.index('baseScreenshot')]
        target_screenshot = row[attributes.index('targetScreenshot')]
        base_platform = row[attributes.index('basePlatform')]
        target_platform = row[attributes.index('targetPlatform')]
        print('%s // %s' % (base_screenshot, target_screenshot))
        print('%s // %s' % (base_platform, target_platform))

        def get_complete_screenshot (screenshot, platform):
            if screenshot.rfind('null') == -1:
                return 'public' + screenshot[1:screenshot.rfind('/')] + '/complete.png'
            else:
                def get_platform_folder (platform):
                    if platform.rfind('iPhone 12 mini') >= 0:
                        return 'iphone12mini'
                    elif platform.rfind('iPhone 12 Pro Max') >= 0:
                        return 'iphone12max'
                    elif platform.rfind('iPhone 12') >= 0:
                        return 'iphone12'
                    else:
                        return 'pixel_xl'

                platform_folder = get_platform_folder(platform)
                website_folder = screenshot[1:screenshot.rfind('/')]

                return ('public%s/results/%s/complete.png' % (website_folder, platform_folder))

        base_screenshot = get_complete_screenshot(base_screenshot, base_platform)
        target_screenshot = get_complete_screenshot(target_screenshot, target_platform)
        print('%s -- %s' % (base_screenshot, target_screenshot))

        try:
            with Image.open(base_screenshot) as base_img, Image.open(target_screenshot) as target_img:
                (base_width, base_height) = base_img.size
                (target_width, target_height) = target_img.size

                baseX = row[attributes.index('baseX')]
                targetX = row[attributes.index('targetX')]
                baseY = row[attributes.index('baseY')]
                targetY = row[attributes.index('targetY')]

                if (baseX == -1 or baseX >= base_width) and (targetX == -1 or targetX >= target_width):
                    outside.append(row)
                elif (baseY == -1 or baseY >= base_height) and (targetY == -1 or targetY >= target_height):
                    outside.append(row)
                else:
                    new_data.append(row)

        except:
            removed_website = removed_website + 1
            print('Removed website')
            outside.append(row)

    print(len(outside))
    print('Removed rows %d' % (removed_website))

    dataset['data'] = new_data

    with open('./data/dataset.unclassified.filtered.arff', 'wt') as new_f:
        new_f.write(arff.dumps(dataset))
