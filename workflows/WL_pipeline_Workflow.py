import os
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import photoZCharacterization
import null_tests

def setupPhotoZChar():
    for i, job in enumerate(photoZCharacterization.jobs):
        pipeline.createSubstream("photoZChar", i, job.pipeline_vars)

def setupCatalogSelectionNullTests():
    for i, null_test in enumerate(null_tests.catalogSelectionNullTests):
        pipeline.createSubstream("catalogSelectionNullTest", i,
                                 null_test.pipeline_vars)

def setupPhotozCharNullTests():
    for i, null_test in enumerate(null_tests.photoZCharNullTests):
        pipeline.createSubstream("photoZCharNullTest", i,
                                 null_test.pipeline_vars)

def setupTBinningNullTests():
    for i, null_test in enumerate(null_tests.tBinningNullTests):
        pipeline.createSubstream("tBinningNullTest", i,
                                 null_test.pipeline_vars)

def setupDNdZInferenceNullTests():
    for i, null_test in enumerate(null_tests.dNdZInferenceNullTests):
        pipeline.createSubstream("dNdZInferenceNullTest", i,
                                 null_test.pipeline_vars)

def setup2PCFEstimateNullTests():
    for i, null_test in enumerate(null_tests.twoPCFEstimateNullTests):
        pipeline.createSubstream("2PCFEstimateNullTest", i,
                                 null_test.pipeline_vars)

def setupCovarianceModelNullTests():
    for i, null_test in enumerate(null_tests.covarianceModelNullTests):
        pipeline.createSubstream("covarianceModelNullTest", i,
                                 null_test.pipeline_vars)
