import pandas as pd
from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel, MultiLabelClassificationArgs
from sys import argv
from sklearn.metrics import accuracy_score, f1_score
import torch
import os

print(torch.cuda.is_available())

print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())


print(torch.cuda.get_device_name(0))


torch.cuda.empty_cache()


modelos = [
    #############################################################################################################{'model_type': 'xlmroberta', 'model_name': 'xlm-roberta-base', 'num_labels': 2},  # XLM Roberta
    {'model_type': 'bert', 'model_name': 'finiteautomata/beto-sentiment-analysis', 'num_labels': 3},  # BETO  /   Aprobado
    {'model_type': 'distilbert', 'model_name': 'distilbert-base-multilingual-cased', 'num_labels': 2}, # Distil Multi  /  Aprobado
    {'model_type': 'bert', 'model_name': 'bert-base-multilingual-cased', 'num_labels': 2}, # BERT Multi  /  Aprobado
    {'model_type': 'roberta', 'model_name': 'bertin-project/bertin-base-xnli-es', 'num_labels': 3},  # Bertin   /  Aprobado
    {'model_type': 'distilbert', 'model_name': 'philschmid/distilbert-base-multilingual-cased-sentiment-2', 'num_labels': 3},  # Distil_2  /   Aprobado 
    {'model_type': 'roberta', 'model_name': 'edumunozsala/RuPERTa_base_sentiment_analysis_es', 'num_labels': 2}, # Ruperta  /  Aprobado
]


def fun_bertin(label):
    if label == 0:
        return 2
    elif label == 1:
        return 0
    else:
        return 1

#############################
def fun_distilbert_2(label):
    if label == 0:
        return 0
    elif label == 1:
        return 2
    else:
        return 1
############################

def train_model(model_type, model_name, train_df_v1, test_df_v1, num_labels):

    train_df = train_df_v1.copy()
    test_df = test_df_v1.copy()

    mn = ""
    if model_type == "bert":
        if model_name == "finiteautomata/beto-sentiment-analysis":
            mn = "beto"
        else:
            mn = "bert"
    elif model_type == "distilbert":
        if model_name == "distilbert-base-multilingual-cased":
            mn = "distilbert"
        else:
            mn = "distilbert_2"
    elif model_type == "roberta":
        if model_name == "bertin-project/bertin-base-xnli-es": 
            mn = "bertin"
        else:
            mn = "ruperta"

    total_epochs = 15


    if mn == "beto":
        df2 = train_df["labels"].replace(1,2)
        train_df["labels"] = df2
        df2 = test_df["labels"].replace(1,2)
        test_df["labels"] = df2
    elif mn == "bertin":
        train_df["labels"] = train_df["labels"].apply(fun_bertin)
        test_df["labels"] = test_df["labels"].apply(fun_bertin)
    elif mn == "distilbert_2": 
        train_df["labels"] = train_df["labels"].apply(fun_distilbert_2)
        test_df["labels"] = test_df["labels"].apply(fun_distilbert_2)
    else:
        num_labels = 2


    train_df = train_df.iloc[:10]
    test_df = test_df.iloc[:10]


    cuda_available = torch.cuda.is_available()

    currently_epoch = 0

    train_args = {
        'output_dir': f'../Outputs/Resultado-{model_type}-{model_name}-outputs/',
        'fp16': False,

        'max_seq_length': 256,
        'num_train_epochs': total_epochs,
        'train_batch_size': 4,
        'eval_batch_size': 4,
        'gradient_accumulation_steps': 1,
        'learning_rate': 5e-5,
        'save_steps': -1,

        'reprocess_input_data': False,
        "save_model_every_epoch": False,
        'overwrite_output_dir': True,
        'no_cache': True,

        'use_multiprocessing': False,
        'use_multiprocessing_for_evaluation': False,

        'use_early_stopping': True,
        'early_stopping_patience': 3,
        'manual_seed': 4,
    }


    print(cuda_available)
    print(model_type)
    print(model_name)

    model = ClassificationModel(model_type, model_name, num_labels=num_labels, args=train_args, use_cuda=cuda_available)

    acc_score,f1_label = 0,0


    model.train_model(train_df,
    #f1=sklearn.metrics.f1_score, 
    #acc=sklearn.metrics.accuracy_score
    )

    X_test = [test_df["text"][idx] for idx in test_df.index]


    y_predict, raw_outputs = model.predict(X_test)

    y_test = test_df["labels"] 

    acc = accuracy_score(y_predict, y_test)
    if mn != "beto" and mn != "bertin" and mn != "distilbert_2":
        f1 = f1_score(y_predict, y_test)       


    with open(f'../Outputs/Resultado-{model_type}-{model_name}-outputs/Scores-{mn}.txt', "w") as txt_file:
        txt_file.write(f"acc: {acc}\n")
        if mn != "beto" and mn != "bertin" and mn != "distilbert_2":
            txt_file.write(f"f1: {f1}")

    del model
    del train_df
    del test_df


#train_df = pd.read_csv(f"../train.csv")
#test_df = pd.read_csv(f"../test.csv")

train_df = pd.read_csv(f"../gimnasio.csv")
test_df = pd.read_csv(f"../partido.csv")

for model in modelos:
    model_type = model['model_type']
    model_name = model['model_name']
    num_labels = model['num_labels']
    train_model(model_type, model_name, train_df, test_df, num_labels)