"""
A file to play around and run necessary tests
"""
import test

dataset_tester = test.test_dataloader.TestBDD100k()
model_tester = test.test_model_output.TestModel()
postprocess_tester = test.test_postprocess.TestPostprocess()

if __name__=="__main__":
    # dataset_tester.test_dataset_scaling_and_reversion()
    postprocess_tester.test_postprocess_on_dataset_output()
    # postprocess_tester.test_postprocess_on_simple_pretrained_model()
    # model_tester.test_model_ouptut()
    pass