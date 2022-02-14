from pipeline.extractor.xbi_extractor import XBIExtractor
from pipeline.extractor.crosscheck_extractor import CrossCheckExtractor
from pipeline.extractor.browserbite_extractor import BrowserbiteExtractor
from pipeline.extractor.browserninja import *
from pipeline.extractor.browserninja.font_family_extractor import FontFamilyExtractor
from pipeline.extractor.browserninja.image_moments_extractor import ImageMomentsExtractor
from pipeline.extractor.browserninja.relative_position_extractor import RelativePositionExtractor

def get_extractor(name):
    features = []
    extractor = None

    if name == 'browserbite':
        extractor = BrowserbiteExtractor(class_attr)
    elif name == 'crosscheck':
        extractor = CrossCheckExtractor(class_attr)
    elif name == 'browserninja1':
        extractor = BrowserNinjaCompositeExtractor(class_attr,
            extractors=[
                ComplexityExtractor(),
                ImageComparisonExtractor(),
                SizeViewportExtractor(),
                VisibilityExtractor(),
                PositionViewportExtractor(),
            ])
    elif name == 'browserninja2':
        features = [ 'emd', 'ssim', 'mse', 'ncc', 'sdd', 'missmatch', 'psnr',
                     'base_centroid_x', 'base_centroid_y', 'base_orientation',
                     'target_centroid_x', 'target_centroid_y', 'target_orientation',
                     'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
                     'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
                     'target_bin1', 'target_bin2', 'target_bin3',
                     'target_bin4', 'target_bin5', 'target_bin6',
                     'target_bin7', 'target_bin8', 'target_bin9', 'target_bin10' ]
        extractor = BrowserNinjaCompositeExtractor(class_attr,
            extractors=[
                ComplexityExtractor(),
                ImageComparisonExtractor(),
                SizeViewportExtractor(),
                VisibilityExtractor(),
                PositionViewportExtractor(),
                RelativePositionExtractor(),
                PlatformExtractor(),
                ImageMomentsExtractor()
            ])
    else:
        features = [ 'emd', 'ssim', 'mse', 'ncc', 'sdd', 'missmatch', 'psnr',
                     'base_centroid_x', 'base_centroid_y', 'base_orientation',
                     'target_centroid_x', 'target_centroid_y', 'target_orientation',
                     'base_bin1', 'base_bin2', 'base_bin3', 'base_bin4', 'base_bin5',
                     'base_bin6', 'base_bin7', 'base_bin8', 'base_bin9', 'base_bin10',
                     'target_bin1', 'target_bin2', 'target_bin3',
                     'target_bin4', 'target_bin5', 'target_bin6',
                     'target_bin7', 'target_bin8', 'target_bin9', 'target_bin10' ]
        extractor = BrowserNinjaCompositeExtractor(class_attr,
            extractors=[
                ComplexityExtractor(),
                ImageComparisonExtractor(),
                SizeViewportExtractor(),
                VisibilityExtractor(),
                PositionViewportExtractor(),
                RelativePositionExtractor(),
                PlatformExtractor(),
                FontFamilyExtractor(),
                ImageMomentsExtractor()
            ])

    return (extractor, features)
