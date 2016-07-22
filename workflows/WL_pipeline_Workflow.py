import null_tests

def setupCatalogSelectionNullTests():
    for i, null_test in enumerate(null_tests.catalogSelectionNullTests):
        pipeline.createSubstream("catalogSelectionNullTest", i,
                                 null_test.pipeline_vars)

def setupPhotoZChar():
    for i, job in enumerate(null_tests.photoZCharacterizationJobs):
        pipeline.createSubstream("photoZChar", i, job.pipeline_vars)

def setupPhotoZCharNullTests():
    for i, null_test in enumerate(null_tests.PhotoZCharNullTests):
        pipeline.createSubstream("PhotoZCharNullTest", i,
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
    for i, null_test in enumerate(null_tests.2PCFEstimateNullTests):
        pipeline.createSubstream("2PCFEstimateNullTest", i,
                                 null_test.pipeline_vars)

def setupCovarianceModelNullTests():
    for i, null_test in enumerate(null_tests.CovarianceModelNullTests):
        pipeline.createSubstream("covarianceModelNullTest", i,
                                 null_test.pipeline_vars)
