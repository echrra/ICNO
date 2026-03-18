from model import  ICNO_Structured_Mesh_2D

def get_model(args):
    model_dict = {

        'ICNO_Structured_Mesh_2D': ICNO_Structured_Mesh_2D,

    }
    return model_dict[args.model]
