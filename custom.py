import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import cv2
from mrcnn.visualize import display_instances
import matplotlib.pyplot as plt
import imgaug

# Root directory of the project
ROOT_DIR = r"/content/FYP-Test"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")



class CustomConfig(Config):
    """Configuration for training on the custom  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2  # Background + phone,laptop and mobile

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 5

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9

############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset):
        """Load a subset of the Dog-Cat dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("object", 1, "Bridal Kameez")
        self.add_class("object", 2, "Bridal Dupatta")
        self.add_class("object", 3, "Bridal Choli")
        self.add_class("object", 4, "Bridal Lehenga")
        self.add_class("object", 5,"Wedding Guest Lehenga")
        self.add_class("object", 6,"Wedding Guest Dupatta")
        self.add_class("object", 7,"Wedding Guest Choli")
        self.add_class("object", 8,"Bridal Maxi")
        self.add_class("object", 9,"Wedding Guest Sharara")
        self.add_class("object", 10,"Bridal Peplum")
        self.add_class("object", 11,"Wedding Guest Maxi")
        self.add_class("object", 12,"Wedding Guest Sari")
        self.add_class("object", 13,"Wedding Guest Kaftan")
        self.add_class("object", 14,"Wedding Guest Trousers")
        self.add_class("object", 15,"Wedding Guest Kurta")
        self.add_class("object", 16,"Wedding Guest Gharara")
        self.add_class("object", 17,"Wedding Guest Jacket")
        self.add_class("object", 18,"Wedding Guest Angrakha")
        self.add_class("object", 19,"Wedding Guest Shalwar")
        self.add_class("object", 20,"Wedding Guest Frock")
        self.add_class("object", 21,"Wedding Guest Bell Bottom")
        self.add_class("object", 22,"Wedding Guest Straight Pant")
        self.add_class("object", 23,"Wedding Guest Palazzo Pants")
        self.add_class("object", 24,"Semi-Formal Dupatta")
        self.add_class("object", 25,"Semi-Formal Kurta")
        self.add_class("object", 26,"Semi-Formal Lehenga")
        self.add_class("object", 27,"Semi-Formal Chooridar")
        self.add_class("object", 28,"Semi-Formal Culottes")
        self.add_class("object", 29,"Semi-Formal Jacket")
        self.add_class("object", 30,"Semi-Formal Maxi")
        self.add_class("object", 31,"Semi-Formal Angrakha")
        self.add_class("object", 32,"Semi-Formal Straight Pant")
        self.add_class("object", 33,"Semi-Formal Shalwar")
        self.add_class("object", 34,"Semi-Formal Trousers")
        self.add_class("object", 35,"Semi-Formal Gharara")
        self.add_class("object", 36,"Semi-Formal Culottes")                      
        self.add_class("object", 37,"Semi-Formal Bell Bottom")                                
        self.add_class("object", 38,"Semi-Formal Sari")   
        self.add_class("object", 39,"Semi-Formal Palazzo Pants")
        self.add_class("object", 40,"Bridal Kurta")      
        self.add_class("object", 41,"Everyday Casual Kurta")
        self.add_class("object", 42,"Everyday Casual Dupatta")
        self.add_class("object", 43,"Everyday Casual Bell Bottom") 
        self.add_class("object", 44,"Everyday Casual Straight Pant")
        self.add_class("object", 45,"Everyday Casual Culottes")
        self.add_class("object", 46,"Everyday Casual Shalwar")
        self.add_class("object", 47,"Everyday Casual Trousers") 
        self.add_class("object", 48,"Everyday Casual Frock") 
        self.add_class("object", 49,"Everyday Casual Gharara") 
        self.add_class("object", 50,"Everyday Casual Chooridar") 
        self.add_class("object", 51,"Everyday Casual Jacket")
        self.add_class("object", 52,"Everyday Casual Maxi")
        self.add_class("object", 53,"Everyday Casual Peplum")   
        self.add_class("object", 54,"Everyday Casual Palazzo Pants") 
        self.add_class("object", 55,"Everyday Casual Lehenga") 
        self.add_class("object", 56,"Semi-Formal Saree") 
        self.add_class("object", 57, "Bridal Long Shirt")
        self.add_class("object", 58, "Bridal Short Shirt")


        # Train or validation dataset?
        
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        if subset == "train":
            annotations1 = json.load(open(r'/content/FYP-Test/dataset/train/train.json'))
        elif subset == "val":
            annotations1 = json.load(open(r'/content/FYP-Test/dataset/val/val.json'))
        elif subset == "test":
            annotations1 = json.load(open(r'/content/FYP-Test/dataset/test/test.json'))
            
        annotations1 = json.load(open('D:/MaskRCNN-main/Dataset/train/via_project.json'))
        # print(annotations1)
        annotations = list(annotations1.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            # print(a)
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()] 
            objects = [s['region_attributes']for s in a['regions'].values()]
           
           
     
            num_ids=[]
            

            for n in objects:

                try:

                    if n['name']== "Bridal Kameez":
                        num_ids.append(1)

                    elif n['name']=="Bridal Dupatta":
                         num_ids.append(2)
                    
                    elif n['name']=="Bridal Choli":
                         num_ids.append(3)

                    elif n['name']=="Bridal Lehenga":
                         num_ids.append(4)
                    
                    elif n['name']=="Wedding Guest Lehenga":
                         num_ids.append(5)
                    
                    elif n['name']=="Wedding Guest Dupatta":
                         num_ids.append(6)

                    elif n['name']=="Wedding Guest Choli":
                         num_ids.append(7)
                    
                    elif n['name']=="Bridal Maxi":
                         num_ids.append(8)

                    elif n['name']=="Wedding Guest Long Shirt":
                         num_ids.append(9)
                    
                    elif n['name']=="Bridal Peplum":
                         num_ids.append(10)
                    
                    elif n['name']=="Wedding Guest Maxi":
                         num_ids.append(11)
                         
                    elif n['name']=="Wedding Guest Sari":
                         num_ids.append(12)

                    elif n['name']=="Wedding Guest Kaftan":
                         num_ids.append(13)
                    
                    elif n['name']=="Wedding Guest Trousers":
                         num_ids.append(14)

                    elif n['name']=="Wedding Guest Kurta":
                         num_ids.append(15)

                    elif n['name']=="Wedding Guest Gharara":
                         num_ids.append(16)
                    
                    elif n['name']=="Wedding Guest Jacket":
                         num_ids.append(17)
                    
                    elif n['name']=="Wedding Guest Angrakha":
                         num_ids.append(18)
                    
                    elif n['name']=="Wedding Guest Shalwar":
                         num_ids.append(19)
                    
                    elif n['name']=="Wedding Guest Frock":
                         num_ids.append(20)
                    
                    elif n['name']=="Wedding Guest Bell Bottom":
                         num_ids.append(21)
                    
                    elif n['name']=="Wedding Guest Straight Pant":
                         num_ids.append(22)
                    
                    elif n['name']=="Wedding Guest Palazzo Pants":
                         num_ids.append(23)
                    
                    elif n['name']=="Semi-Formal Dupatta":
                         num_ids.append(24)
                    
                    elif n['name']=="Semi-Formal Kurta":
                         num_ids.append(25)
                    
                    elif n['name']=="Semi-Formal Lehenga":
                         num_ids.append(26)
                    
                    elif n['name']=="Semi-Formal Chooridar":
                         num_ids.append(27)
                    
                    elif n['name']=="Semi-Formal Culottes":
                         num_ids.append(28)
      
                    elif n['name']=="Semi-Formal Jacket":
                         num_ids.append(29)
                    
                    elif n['name']=="Semi-Formal Maxi":
                         num_ids.append(30)
                    
                    elif n['name']=="Semi-Formal Angrakha":
                         num_ids.append(31)
                    
                    elif n['name']=="Semi-Formal Straight Pant":
                         num_ids.append(32)
                    
                    elif n['name']=="Semi-Formal Shalwar":
                         num_ids.append(33)
                    
                    elif n['name']=="Semi-Formal Trousers":
                         num_ids.append(34)
                    
                    elif n['name']=="Semi-Formal Gharara":
                         num_ids.append(35)
                    
                    elif n['name']=="Semi-Formal Culottes":
                         num_ids.append(36)
                    
                    elif n['name']=="Semi-Formal Bell Bottom":
                         num_ids.append(37)
        
                    elif n['name']=="Semi-Formal Sari":
                         num_ids.append(38)   
                    
                    elif n['name']=="Semi-Formal Palazzo Pants":
                         num_ids.append(39)
                    
                    elif n['name']=="Bridal Kurta":
                         num_ids.append(40)
                    
                    elif n['name']=="Everyday Casual Kurta":
                         num_ids.append(41)

                    elif n['name']=="Everyday Casual Dupatta":
                         num_ids.append(42)
                    
                    elif n['name']=="Everyday Casual Bell Bottom":
                         num_ids.append(43)
                    
                    elif n['name']=="Everyday Casual Straight Pant":
                         num_ids.append(44)
                    
                    elif n['name']=="Everyday Casual Culottes":
                         num_ids.append(45)

                    elif n['name']=="Everyday Casual Shalwar":
                         num_ids.append(46)
                    
                    elif n['name']=="Everyday Casual Trousers":
                         num_ids.append(47)
                    
                    elif n['name']=="Everyday Casual Frock":
                         num_ids.append(48)
                    
                    elif n['name']=="Everyday Casual Gharara":
                         num_ids.append(49)
                    
                    elif n['name']=="Everyday Casual Chooridar":
                         num_ids.append(50)
                    
                    elif n['name']=="Everyday Casual Jacket":
                         num_ids.append(51)
                    
                    elif n['name']=="Everyday Casual Maxi":
                         num_ids.append(52)
                    
                    elif n['name']=="Everyday Casual Peplum":
                         num_ids.append(53)
                    
                    elif n['name']=="Everyday Casual Palazzo Pants":
                         num_ids.append(54)
                    
                    elif n['name']=="Everyday Casual Lehenga":
                         num_ids.append(55)
                    
                    elif n['name']=="Semi-Formal Saree":
                         num_ids.append(56)
                    
                    elif n['name']=="Bridal Long Shirt":
                         num_ids.append(57)

                    elif n['name']=="Bridal Short Shirt":
                         num_ids.append(58)
                      
                except:
                    
                    print("Not found:",n['name'])

          
            
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",  ## for a single class just add the name here
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons,
                num_ids=num_ids
                )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a Dog-Cat dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        if info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)
        num_ids = info['num_ids']
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
        	rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])

        	mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        # Map class names to class IDs.
        num_ids = np.array(num_ids, dtype=np.int32)
        return mask, num_ids #np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "object":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(r'/content/FYP-Test/dataset', "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(r'/content/FYP-Test/dataset', "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=25,
                layers='heads', #layers='all', 
                augmentation = imgaug.augmenters.Sequential([ 
                imgaug.augmenters.Fliplr(0.5), 
                imgaug.augmenters.Flipud(0.5), 
                imgaug.augmenters.Affine(rotate=(-45, 45)), 
                imgaug.augmenters.Affine(rotate=(-90, 90)), 
                imgaug.augmenters.Affine(scale=(0.5, 1.5)),
                imgaug.augmenters.Crop(px=(0, 10)),
                imgaug.augmenters.Grayscale(alpha=(0.0, 1.0)),
                imgaug.augmenters.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                imgaug.augmenters.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                imgaug.augmenters.Invert(0.05, per_channel=True), # invert color channels
                imgaug.augmenters.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                
                ])
                )
				
				
config = CustomConfig()
model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

weights_path = COCO_WEIGHTS_PATH
        # Download weights file
if not os.path.exists(weights_path):
  utils.download_trained_weights(weights_path)

model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])

train(model)			