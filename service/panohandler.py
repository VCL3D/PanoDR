from handler_panodr import panodrHandler 
import logging

_service = panodrHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
        logging.info("initialition succeded")

    if data is None:
        return None
    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data
   