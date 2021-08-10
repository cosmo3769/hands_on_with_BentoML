# bento_packer.py - after prediction service class is constructed, 
# it packages the trained model with the prediction service 
# class "IrisClassifier" defined above, and then saves the 
# IrisClassifier instance to disk in the BentoML format 
# for distribution and deployment, under bento_packer.py 

# import the IrisClassifier class defined above
from bento_service import IrisClassifier
from train import clf

# Create a iris classifier service instance
iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)

# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()