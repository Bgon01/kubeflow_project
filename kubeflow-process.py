# Import Kubeflow packages
from kfp import dsl,compiler
from kfp.dsl import ContainerOp
import json,ast

# initialize the kubeflow pipeline decorator
@dsl.pipeline(name='deep_learning_pipeline', description='Kubeflow pipeline to preprocess, train and deploy a deep learning model')
def youtube_pipeline():
    # initialize the kubeflow pipeline operator for  running the preprocess step 
    preprocess_operator = dsl.ContainerOp(
        name='Pre-process',
        image='gcr.io/imaya-2/deep_learning_project_for_text_detection_in_images:latest',
        command=['sh','-c'],
        arguments=['cd deep_learning_folder && python3 src/preprocessing.py'],
        file_outputs={'data_file': '/deep_learning_folder/input/data/data_file.csv',
                        'char2int': '/deep_learning_folder/input/data/char2int.pkl',
                        'int2char':'/deep_learning_folder/input/data/int2char.pkl'})

    # initialize the kubeflow pipeline operator for  running the training step 
    train_op = dsl.ContainerOp(
    name = f'Training',
    image = 'gcr.io/imaya-2/deep_learning_project_for_text_detection_in_images:latest',
    command=['sh','-c'],
    arguments=[f"echo 'Hello' && cat {preprocess_operator.outputs['data_file']}"])

    # set the caching strategy to 'P0D' for preprocess and training step. this is used for loading a pod with fresh state 
    preprocess_operator.execution_options.caching_strategy.max_cache_staleness = "P0D"
    train_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # Setup the order in which the pipeline has to run
    train_op.after(preprocess_operator)    
    
# call the compile to output the yaml file for the process we defined above
if __name__ == '__main__':
    compiler.Compiler().compile(youtube_pipeline, 'deep_learning_pipeline.yaml')


