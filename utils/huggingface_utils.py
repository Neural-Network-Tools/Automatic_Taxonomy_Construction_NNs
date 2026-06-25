from huggingface_hub import list_models as huggingface_list_models
from huggingface_hub import model_info
from transformers import pipeline
from pathlib import Path
import io
from types import SimpleNamespace

def get_model(searchstring):
    models = huggingface_list_models(search=searchstring, limit=1) #,filter=[task, "pytorch", "safetensors"],sort="downloads")
    models = list(models)
    if len(models) > 0:
        return models[0]
    else: 
        return None

def get_models(searchstring):
    models = huggingface_list_models(search=searchstring, limit=10, sort="downloads") #,filter=[task, "pytorch", "safetensors"],sort="downloads")
    models = list(models)
    return models
def download_model_and_make_source_code(hfmodel,modelname="somemodel",device='cpu',userdatadir='./tmp'):
    # downloads model as pytorch 
    hf_directory = Path(userdatadir)
    hf_directory.mkdir(parents=True, exist_ok=True)
    pipeline_tag = hfmodel.pipeline_tag
    id = hfmodel.id
    pipelinemodel = pipeline(pipeline_tag,model=id)
    if pipeline_tag == 'text-generation':
        generation_kwargs=',generation_kwargs={"use_cache": False}'
    else:
        generation_kwargs=''
    makefile = f"""
from transformers import pipeline

pipeline = pipeline("{pipeline_tag}",model="{id}",torchscript=True,return_dict=False{generation_kwargs})
model = pipeline.model
model.config.use_cache = False
"""

    modelname = modelname.replace('/','_')+'.py'
    #hf_python_file = hf_directory / f'{modelname}.py' 
    #handle = open(hf_python_file, 'w')
    #handle.write(makefile)
    #handle.close()
    return SimpleNamespace(name= modelname,getvalue = lambda: io.BytesIO(makefile.encode('utf-8')).getvalue()) #str(hf_python_file)
    
    