from custom_utils import *
import matplotlib.pyplot as plt
import pdb
labeled_dataloader = get_labeled_data("LabeledData", labeled_annotations, 4, False)
images, labels = next(iter(labeled_dataloader))
pdb.set_trace()