from custom_utils import *
import matplotlib.pyplot as plt
import pdb
# labeled_dataloader = get_labeled_data("LabeledData", labeled_annotations, 4, False)
# images, labels = next(iter(labeled_dataloader))
labeled_annotations = get_annotations('ClimbingHoldDetection-15/train/_annotations.coco.json')
labeled_dataloader = get_labeled_data("labeledData", labeled_annotations, 4, False)
images, labels = next(iter(labeled_dataloader))
pdb.set_trace()