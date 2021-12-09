"""
Handler for invoking service with JSON job description and posting results to specified endpoint.
"""
import logging
import torch
import io
import requests
import json
from abc import ABC
from PIL import Image
import numpy as np
from base_handler import BaseHandler
import cv2
from torchvision import transforms as T
from panorama import *
import urllib.request

logger = logging.getLogger(__name__)

class panodrHandler(BaseHandler, ABC):
    """
    ImageSegmenter handler class. This handler takes a batch of images
    and returns output shape as [N K H W],
    where N - batch size, K - number of classes, H - height and W - width.
    """
    parameter_dict = {}

    def preprocess_image(self, img_data, msk, layout, layout_shape):
        """
        Inputs: RGB image, mask or list of polygon points , RGB layout or list of corners
        Outputs: RGB tensor, mask tensor, layout (one_hot) tensor
        """
        _transform = T.ToTensor()
        images = (_transform(img_data)).to(self.device)
        labels_one_hot = [[0, 0, 1], [1, 0, 0], [0, 1, 0]] #Three(3) classes (Ceiling, Floor, Wall)
        
        if(isinstance(msk, list)):
            msk = np.array(msk)
            mask = np.zeros((layout_shape[0], layout_shape[1]))
            cv2.fillPoly(mask, msk, 1)
            mask = torch.from_numpy(mask).to(self.device)/255
        else:
            mask = (_transform(msk)).to(self.device)

        if(isinstance(layout, list)):
            """
            Input: List of corners
            Output: onehot tensor
            """
            corners = np.array(layout)
            img_l = np.zeros((layout_shape[0],layout_shape[1],layout_shape[2]))
            layout_t, _ = Layout(corners, img_l, img_data.size[0], img_data.size[1])
            semantic_mask = Layout2Semantic(layout_t)
            labels_panorama, _ = getLabels(semantic_mask)
            labels_panorama = convert_3_classes(labels_panorama)
            _layout = one_hot(labels_panorama.squeeze_(0).permute(2,0,1), 3)
            
        else:
            _layout = (torch.from_numpy(layout).to(self.device).permute(2,0,1))


        return images, mask, _layout

    def preprocess(self, data):

        logger.info("******************* JsonMsgHandler: preprocess  file*********************")

        json_obj = {}
        for row in data:
            logger.info("******************* Row is : {} ".format(row))
            json_obj = row.get("data") or row.get("body")
            self.parameter_dict = dict(json_obj)
        logger.info(" {} ".format(row))
        #read out information needed from the json:
        image_url = ""
        logger.info("{}".format(str(json_obj)))
        json_obj = dict(json_obj)

        self.device = torch.device("cuda:" + str(json_obj['DataInputs']['gpu_id']) if torch.cuda.is_available() else "cpu")
        target_shape = json_obj["Source"]["shape"]
        image_url = json_obj['DataInputs']['rgb']
        mask_url = json_obj['DataInputs']['mask']
        layout_url = json_obj['DataInputs']['layout']
        layout_shape = json_obj['DataInputs']['layout_shape']
        use_layout_serv = json_obj['LayoutService']['use_layout']
        r = ""
        msk = ""
        l = ""
        try:
            # send request to the webserver to get the image specified in the json
            r = requests.get(image_url, timeout=0.5)
            if isinstance(mask_url, str):
                msk = requests.get(mask_url, timeout=0.5)
        except requests.exceptions.RequestException as e:
            logger.warning("*   RequestException: " + str(e))

        image = []
        mask = []
        layout = []
        logger.info("******************* Status code =  {} ".format(r.status_code)) 
        if r.status_code == 200:
            logger.info("******************* Status code =  {} ".format(r.status_code))
            image = (Image.open(io.BytesIO(r.content))).convert("RGB")
            image.resize((target_shape[2], target_shape[1]), Image.BICUBIC)
            if isinstance(mask_url, str):
                mask = Image.open(io.BytesIO(msk.content))
                mask.resize((target_shape[2], target_shape[1]), Image.NEAREST)
            elif isinstance(mask_url, list):
                mask = json_obj['DataInputs']['mask']
            else:
                logger.warning("********Exception: Invalid mask output********")

            # Checking whether user selects to load the layout from its corresponding service
            if use_layout_serv == True:
                layout_url = json_obj['LayoutService']['layout_url']
                try:
                    # send request to the webserver to get the image specified in the json
                    r = requests.get(layout_url, timeout=0.5)

                except requests.exceptions.RequestException as e:
                    logger.warning("*   RequestException: " + str(e))

                layout = Image.open(io.BytesIO(r.content))
                layout = np.array(layout, dtype=np.float32)

            else:

                if isinstance(layout_url, str):
                    layout = urllib.request.urlopen(layout_url)
                    layout = np.loadtxt(layout).tolist()
                elif isinstance(layout_url, list):
                    layout = json_obj['DataInputs']['layout']
                else:
                    logger.warning("********Exception: Invalid layout input********")

        del r,msk,l

        return torch.stack(self.preprocess_image(image, mask, layout, layout_shape))

    def postprocess_image(self, data):
        output = []
        img = T.ToPILImage(mode='RGB')(data[0].squeeze_(0))

        img_raw = T.ToPILImage(mode='RGB')(data[2].squeeze_(0))

        layout = data[1].squeeze_(0).permute(1,2,0).detach().numpy()
        output =([img,layout,img_raw])
        return output

    def postprocess(self, data):

        logger.info("******************* JsonMsgHandler: postprocess *********************")
        data = self.postprocess_image(data) 

        url_to_post = ""
        url_to_post = self.parameter_dict['Parameters']['url']

        #DR result
        img = data[0]
        out_file = io.BytesIO()
        img.save(out_file, format='png')
        out_file.seek(0)

        #DR raw result
        img_raw = data[2]
        out_file_raw = io.BytesIO()
        img_raw.save(out_file_raw, format='png')
        out_file_raw.seek(0)
        #Layout 
        layout = data[1]
        layout = Image.fromarray((layout* 255).astype(np.uint8))

        out_file_layout = io.BytesIO()
        layout.save(out_file_layout, format='png')
        out_file_layout.seek(0)

        logger.info("URL to post: {}".format(url_to_post))
        
        json_dict = {
            "id": str("Diminished Scene & Layout"),
            "value": str(url_to_post)

        }

        json_to_return = dict(json_dict)
        try:
            r = requests.post(url_to_post, files = {'image': out_file, 'layout': out_file_layout, 'image_empty': out_file_raw})

        except requests.exceptions.RequestException as e:
            logger.warning("******************* RequestException: " + str(e))

        status_entry = {"status": str(r.status_code)}
        json_to_return.update(status_entry)

        a_list = [json.dumps(json_to_return)]

        logger.warning("***********************Service completed***********************")
        return a_list

    def inference(self, data):
        """
        Internal inference methods
        :param model_input: transformed model input data
        :return: list of inference output in NDArray
        """
        # Do some inference call to engine here and return output
        self.model = self.model.to(self.device)
        model_output = self.model.forward(data)
        return model_output