## MLOps On GCP

This repository contains the code files involved in creating an automated MLOps Pipeline on GCP (Google Cloud Platform).

### Intro:
* Clone the repository
* Use the input folder to add more images to your dataset and also add the image entry to ```data_file.csv```

Once you made the changes, create a new repository and commit the changes. From here on, this will be your source repository. Proceed with the below steps
###### Cloud Build Trigger
* In your GCP concole, create a new cloud build trigger.
* Point the trigger to your source repository

###### Cloud Run 
* In Cloud Run, point the CI/CD server towards you cloud build trigger out
* The output from cloud build will be in Artifacts Registry which holds a docker image.
* Cloud run will provide a endpoint, a HTTPS URL which will serve the flask app that is created
* Add the permission "allUsers" with roles as "Cloud Run Invoker" and save the changes
* Once changes the change reflects, the HTTPS URL will be accessible

This completes the process of deployment. Now the next step is to have a kubeflow pipeline that takes care of the model training part.

###### Kubeflow
* In your cloud console access AI Platforms
* Under the pipeline section of AI Platforms, click ```Create Instances```
* Go through the steps of naming the kubeflow cluster and having granular level access of the machine configuration for your clusters
* Within the created cluster, create a pipeline using the ```deep_learning_pipeline.yaml```.
* At this point you will have control over steps in your pipeline and the graph view of them as well.
* There are two option for running the pipelines, either schedule it at the creation time itself or run it manually.
