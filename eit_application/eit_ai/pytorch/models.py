import os
from abc import ABC, abstractmethod
import logging
from typing import Any
from contextlib import redirect_stdout
from torchinfo import summary
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from glob_utils.directory.utils import (
    append_date_time,
    get_datetime_s,
    get_dir,
    mk_new_dir,
)
import numpy as np
import torch
from eit_ai.pytorch.const import (PYTORCH_LOSS, PYTORCH_MODEL_SAVE_FOLDERNAME,
                                  PYTORCH_OPTIMIZER)
from eit_ai.pytorch.dataset import (DataloaderGenerator,
                                    StdPytorchDatasetHandler)
from eit_ai.train_utils.dataset import AiDatasetHandler
from eit_ai.train_utils.lists import (ListPyTorchLosses,
                                      ListPytorchModelHandlers,
                                      ListPytorchModels, ListPyTorchOptimizers,
                                      get_from_dict)
from eit_ai.train_utils.metadata import MetaData, reload_metadata
from eit_ai.train_utils.models import (MODEL_SUMMARY_FILENAME,
                                       AiModelHandler, ModelNotDefinedError,
                                       WrongLearnRateError, WrongMetricsError)
from genericpath import isdir, isfile
from torch import nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)
writer = SummaryWriter()

class TypicalPytorchModel(ABC):
    """Define a standard pytorch Model
    """    
    
    net:nn.Module=None
    name:str=None
    def __init__(self, metadata: MetaData) -> None:
        super().__init__()
        self._set_layers(metadata)
        self.net.apply(self.init_weights)
    
    @abstractmethod
    def _set_layers(self, metadata:MetaData)-> None:
        """define the layers of the model and the name

        Args:
            metadata (MetaData): [description]
        """ 
    
    def prepare(self, op:torch.optim.Optimizer, loss):
        self.optimizer= op
        self.loss= loss
        
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        # logger.debug(f'foward, {x.shape=}')
        return self.net(x)

    def init_weights(self,m):
        if type(m) == nn.Linear or type(m) == nn.Conv1d:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


    def train_single_epoch(self, dataloader:DataLoader) -> Any:
        self.net.train()
        # logger.debug(f'run_single_epoch')
        for idx, data_i in enumerate(dataloader):
            # logger.debug(f'Batch #{idx}')
            train_loss = 0
            inputs, labels = data_i
            inputs = inputs.to(device=0)
            labels = labels.to(device=0)
            
            # zero gradients for every batch
            self.optimizer.zero_grad()

            y_pred = self.net(inputs)
            
            #loss
            loss_value = self.loss(y_pred, labels)
            loss_value.backward()

            # adjust learning weights
            self.optimizer.step() 
            
            train_loss += loss_value.item()

        return train_loss / len(dataloader)
    
    def val_single_epoch(self, dataloader: DataLoader) -> Any:
        # logger.debug(f'run_single_validation_epoch')
        val_loss = 0
        with torch.no_grad():
            for idx, data_i in enumerate(dataloader):
                
                inputs, labels = data_i
                inputs = inputs.to(device=0)
                labels = labels.to(device=0)
                y_pred = self.net(inputs)
                
                loss_value = self.loss(y_pred, labels)
                val_loss += loss_value.item()
        return val_loss / len(dataloader) 

    def get_name(self)->str:
        """Return the name of the model/network

        Returns:
            str: specific name of the model/network
        """        
        return self.name

    def get_net(self):
        return self.net

    def predict(self, x_pred: np.ndarray)->np.ndarray:
        """[summary]
        predict the new x
        """
        return self.net(torch.Tensor(x_pred)).detach().numpy()

class StdPytorchModel(TypicalPytorchModel):

    def _set_layers(self, metadata:MetaData)-> None:
        in_size=metadata.input_size
        out_size=metadata.output_size
        self.name= "MLP with 3 layers"
        self.net = nn.Sequential(nn.Linear(in_size,1024),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(1024, 128),
                                nn.ReLU(True),
                                # nn.Dropout(0.5),
                                nn.Linear(128, 1024),
                                nn.ReLU(True),
                                nn.Dropout(0.5),
                                nn.Linear(1024, out_size),
                                nn.Sigmoid()
                                )
       
        self.net.to(device=0)

class Conv1dNet(TypicalPytorchModel):
    
    def _set_layers(self, metadata: MetaData) -> None:

        out_size=metadata.output_size
        self.name = "1d CNN"
        self.net = torch.nn.Sequential(nn.Conv1d(in_channels= 1, out_channels= 18, kernel_size=8, stride=1, padding="same"),
                                       nn.BatchNorm1d(18),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=1),
                                       nn.Conv1d(18, 12, kernel_size=8, stride=1, padding="same"),
                                       nn.BatchNorm1d(12),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=1),
                                       nn.Conv1d(12, 6, kernel_size=16, stride=1, padding="same"),
                                       nn.BatchNorm1d(6),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=1),
                                       nn.Conv1d(6, 4, kernel_size=16, stride=1, padding="same"),
                                       nn.BatchNorm1d(4),
                                       nn.ReLU(True),
                                       nn.MaxPool1d(kernel_size=2, stride=1),
                                       nn.Flatten(),
                                       nn.Linear(1008, 2048),
                                       nn.ReLU(True),
                                       nn.Dropout(0.5),
                                       nn.Linear(2048, out_size),
                                       nn.Sigmoid()
                                    )
        self.net.to(device=0)    

class AutoEncoder(TypicalPytorchModel):  
      
    def _set_layers(self, metadata: MetaData) -> None:

        in_size = metadata.input_size
        out_size=metadata.output_size
        self.name = "AutoEncoder"
        self.net = torch.nn.Sequential()
        
        encoder = nn.Sequential(
            nn.Linear(in_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 16),  
            )
        
        decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(True),
            nn.Linear(64, 256),
            )

        dense = nn.Sequential(
            nn.Linear(256, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, out_size),
            nn.Sigmoid(),
        )
        
        
        self.net.add_module('encoder', encoder)
        self.net.add_module('decoder', decoder)
        self.net.add_module('dense_layer', dense)
        
        self.net.to(device=0)
        
################################################################################
# Std PyTorch ModelManager
################################################################################
class StdPytorchModelHandler(AiModelHandler):

    def _define_model(self, metadata:MetaData)-> None:

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        logger.info(f"Device Cuda: {device}")

        model_cls=get_from_dict(
            metadata.model_type, PYTORCH_MODELS, ListPytorchModels)
        self.model=model_cls(metadata)
        self.name= self.model.get_name()

    def _get_specific_var(self, metadata:MetaData)-> None:   
        self.specific_var['optimizer'] = get_pytorch_optimizer(metadata, self.model.get_net())
        self.specific_var['loss'] = get_pytorch_loss(metadata)
        if not isinstance(metadata.metrics, list):
            raise WrongMetricsError(f'Wrong metrics type: {metadata.metrics}')
        self.specific_var['metrics'] = metadata.metrics   

    def _prepare_model(self)-> None:
        self.model.prepare( 
            self.specific_var['optimizer'],
            self.specific_var['loss']  )

    

    def train(self, dataset: AiDatasetHandler, metadata: MetaData) -> None:
        gen = DataloaderGenerator()
        train_dataloader = gen.make(dataset, 'train', metadata=metadata)
        val_dataloader = gen.make(dataset, 'val', metadata=metadata)
        logger.info(f'Training Start - total {metadata.epoch} epoch')

        # 创建一个字典来保存每轮的损失值
        results = {
            "epoch": [],
            "train_loss": [],
            "val_loss": []
        }

        for epoch in range(metadata.epoch):
            train_loss = self.model.train_single_epoch(train_dataloader)
            val_loss = self.model.val_single_epoch(val_dataloader)

            logger.info(f'epoch #{epoch + 1}/{metadata.epoch} ')
            logger.info(f'train_loss = {train_loss}, val_loss = {val_loss}')

            # 将每一轮的损失值保存到字典中
            results["epoch"].append(epoch + 1)
            results["train_loss"].append(train_loss)
            results["val_loss"].append(val_loss)

            # 记录到 TensorBoard
            writer.add_scalar("training_loss", train_loss, epoch + 1)
            writer.add_scalar("val_loss", val_loss, epoch + 1)
        
        self.time_stamps = get_datetime_s()
        filename = f'training_results_{self.time_stamps}.xlsx'

        folder_path = 'E:/Chen/EIT/eit_ai/VALandtrain'  
        os.makedirs(folder_path, exist_ok=True)  # 如果文件夹不存在，则创建
            # 组合文件夹路径和文件名
        file_path = os.path.join(folder_path, filename)

             # 将结果转换为DataFrame并保存到指定路径的Excel文件
        df = pd.DataFrame(results)
        df.to_excel(file_path, index=False)  # 不需要索引列
        writer.close()

    def predict(
        self,
        X_pred:np.ndarray,
        metadata:MetaData,
        **kwargs)->np.ndarray:

        # X_pred preprocess if needed
        # if X_pred.shape[0]==1:
        #     return self.model.predict(X_pred)
        # else:
        #     res = np.array([])
        #     for i in range(X_pred.shape[0]):
        #         pred = self.model.predict(X_pred[i])
        #         res = np.append(res, pred)
        #     return res  
        return self.model.predict(X_pred)

    def save(self, metadata:MetaData)-> str: 
        
        return save_pytorch_model(self.model.net, dir_path=metadata.dir_path, save_summary=metadata.save_summary)

    def load(self, metadata:MetaData)-> None:
        model_cls=get_from_dict(
        metadata.model_type, PYTORCH_MODELS, ListPytorchModels)
        
        self.model = model_cls(metadata)
        
        self.model.net= load_pytorch_model(dir_path=metadata.dir_path)
        self.model.net.eval()
        

################################################################################
# common methods
################################################################################

def assert_pytorch_model_defined(model:Any)-> nn.Module:
    """allow to react if model not  defined

    Args:
        model (Any): [description]

    Raises:
        ModelNotDefinedError: [description]

    Returns:
        pytorch.models.Model: [description]
    """    
    if not isinstance(model, nn.Module):
        raise ModelNotDefinedError(f'Model has not been correctly defined: {model}')
    return model


def get_pytorch_optimizer(metadata:MetaData, net:nn.Module)-> torch.optim.Optimizer:

    if not metadata.optimizer:
        metadata.optimizer=list(PYTORCH_OPTIMIZER.keys())[0].value

    op_cls=get_from_dict(
        metadata.optimizer, PYTORCH_OPTIMIZER, ListPyTorchOptimizers)

    if metadata.learning_rate:
        if metadata.learning_rate >= 1.0:
            raise WrongLearnRateError(f'Wrong learning rate type (>= 1.0): {metadata.learning_rate}') 
        return op_cls(net.parameters(), lr= metadata.learning_rate)
    
    logger.warning('Learningrate has been set to 0.001!!!')
    return op_cls(net.parameters(), lr=0.001)
        

def get_pytorch_loss(metadata:MetaData)->nn.modules.loss:

    if not metadata.loss:
        metadata.loss=list(PYTORCH_LOSS.keys())[0].value

    loss_cls=get_from_dict(metadata.loss, PYTORCH_LOSS, ListPyTorchLosses)
    return loss_cls()

def save_pytorch_model(net:nn.Module, dir_path:str='', save_summary:bool=False)-> str:
    """Save a pytorch model, additionnaly can be the summary of the model be saved"""
    if not isdir(dir_path):
        dir_path=os.getcwd()
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    
    torch.save(net, model_path)

    logger.info(f'PyTorch model saved in: {model_path}')

    if save_summary:
    
        summary_path= os.path.join(dir_path, MODEL_SUMMARY_FILENAME)
        with open(summary_path, 'w') as f:
            with redirect_stdout(f):
                summary(net, input_size=(32000, 256), device='cpu')
        logger.info(f'pytorch model summary saved in: {summary_path}')
    
    return model_path

def load_pytorch_model(dir_path:str='') -> nn.Module:
    """Load pytorch Model and return it if succesful if not """
    
    metadata = reload_metadata(dir_path=dir_path)
    if not isdir(dir_path):
        logger.info(f'pytorch model loading - failed, wrong dir {dir_path}')
        return
    model_path=os.path.join(dir_path, PYTORCH_MODEL_SAVE_FOLDERNAME)
    if not isfile(model_path):
        logger.info(f'pytorch model loading - failed, {PYTORCH_MODEL_SAVE_FOLDERNAME} do not exist in {dir_path}')
        return None
    try:
        net = torch.load(model_path,map_location='cpu')
        logger.info(f'pytorch model loaded: {model_path}')
        logger.info('pytorch model summary:')
        if metadata.model_type == 'Conv1dNet':
            summary(net, input_size=(metadata.batch_size, 1, metadata.input_size), device='cpu')
        else:
            summary(net, input_size=(metadata.batch_size, metadata.input_size), device='cpu')
        return net.eval()

    except BaseException as e: 
        logger.error(f'Loading of model from dir: {model_path} - Failed'\
                     f'\n({e})')
        return None

################################################################################
# pytorch Models
################################################################################


PYTORCH_MODEL_HANDLERS={
    ListPytorchModelHandlers.PytorchModelHandler: StdPytorchModelHandler, 
}

PYTORCH_MODELS={
    ListPytorchModels.StdPytorchModel: StdPytorchModel, 
    ListPytorchModels.Conv1dNet: Conv1dNet,
    ListPytorchModels.AutoEncoder: AutoEncoder,
}


if __name__ == "__main__":
    import logging

    from eit_ai.raw_data.matlab import MatlabSamples
    from glob_utils.log.log import change_level_logging, main_log
    main_log()
    change_level_logging(logging.DEBUG)
    


    X = np.random.randn(100, 4)
    Y = np.random.randn(100)
    Y = Y[:, np.newaxis]

    raw=MatlabSamples()
    raw.X=X
    raw.Y=Y
    
    # rdn_dataset = PytorchDataset(X, Y)
    md= MetaData()
    md.set_4_dataset()
    dataset = StdPytorchDatasetHandler()
    dataset.build(raw_samples=raw, metadata= md)

    new_model = StdPytorchModelHandler()
    # for epoch in range(50):
    md.set_4_model()
    new_model.train(dataset,md)
