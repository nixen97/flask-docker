import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.dashboard_objs as dashboard
from dash.dependencies import Input, Output
from visualizations import construct_conf_matrices, confusion_matrix
from visualizations import performance_table, plot_training_performances 
from visualizations import viz_embeddings, pred_to_labels
from visualizations import plot_roc_curves, compute_roc, auc_table
from visualizations import compute_avg_of_list, compute_avg_accuracy
from visualizations import compute_avg_conf_matrix, compute_avg_roc_curves
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from flask import Flask
# from flask_cors import CORS
import numpy as np
import json
import os

def launch_dashboard(app):
    app.run_server(debug=True)

def prepare_dashboard(  neural_models_list=['Default1'], 
                        accuracies=[[0]],
                        losses=[[0]],
                        epochs=[0], 
                        all_models=["Default Model"],
                        accuracy_scores=[0],
                        precision_scores=[0],
                        recall_scores=[0],
                        f1_scores=[0],
                        confusion_matrices=[np.array([1,1]), np.array([2,1])],
                        auc=[[0,0],[0,0]],
                        words=["Word1"],
                        embeddings=np.array([0,0,0]),
                        dimensions=3):

                        
    """ this is the function to call from main.py to create the dashboard.
    
    Args:

        neural_models_list (list): List containing the four neural models we have been using.
                                    Needed for plotting training performances.
        accuracies (list of lists): Each list is a list of accuracies during training for each model.
                                    It must respect the order of neural_models_list.
                                    Needed for plotting training performances.
        losses (list of lists): Each list is a list of losses during training for each model. 
                                It must respect the order of neural_models_list.
                                Needed for plotting training performances.
        epochs (list): Number of epochs used during training for each model.
                        It must respect the order of the neural_models_list.
                        Needed for plotting training performances.

        all_models (list): List containing all the used models, excluding the baselines.
                            Needed for confusion matrices, comparison table and ROC curves.

        accuracy_scores (list of lists): accuracies on training for each model

        precision_scores (list of lists): precisions on training for each model

        recall_scores (list of lists): recalls on training for each model

        f1_scores (list of lists): f1 on training for each model

        confusion_matrices (list of arrays): confusion matrices for each model

        auc (list of lists): auc for each model (for class pos and class neg)
        
        words (list): list of words.
                    Needed for embeddings plotting.
        embeddings (np.array): embedding vector for each word in words.
                                Needed for embeddings plotting.
        dimensions (int): Integer. Either two or three. Dimension to visualize the embeddings.
                            Needed for embeddings plotting.
    """


    
    layout = html.Div(
                    children=[html.Div(
                        children=html.Div([
                            dcc.Graph(
                                id='left-top-graph',
                                figure=plot_training_performances(neural_models_list, accuracies, losses, epochs),
                                style={'width': '50%', 'height':'500px', 'display': 'inline-block'}
                            ),
                            dcc.Graph(
                                id='right-top-graph',
                                figure=construct_conf_matrices(confusion_matrices, all_models),
                                style={'width': '50%', 'height':'500px', 'display': 'inline-block'}
                            )
                        ], style={'display': 'inline-block', 'border-width': '0px 0px 0px 0px', 'width':'100%', 'height':'500px'},
                    )),
                    html.Div(
                        children=html.Div([
                            dcc.Graph(
                                id='middle-graph',
                                figure=performance_table(accuracy_scores, precision_scores, recall_scores, f1_scores, all_models),
                                style={'width': '100%', 'height':'500px', 'display': 'inline-block'}
                            )
                        ], style={'display': 'inline-block', 'border-width': '0px 0px 0px 0px', 'width':'100%', 'height':'600px', 'margin-top':'10px', 'margin-bottom': '10px'},
                    )),
                    html.Div(
                        children=html.Div([
                            dcc.Graph(
                                id='bottom-graph',
                                figure=auc_table(auc),
                                style={'width': '100%', 'height':'750px', 'display': 'inline-block'}
                            )
                        ], style={'display': 'inline-block', 'border-width': '0px 0px 0px 0px', 'width':'100%', 'height':'750px', 'margin-top':'10px', 'margin-bottom': '10px'},
                    ))
                    ]
                )

                
    return layout
    #launch_dashboard(app)


def get_data(filename):

    neural_models_list = []
    all_models_list = []
    accuracies = []
    losses = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    conf_matrices = []
    auc = []
    with open(filename) as infile:
        for line in infile:
            model = json.loads(line)
            if model['name'] == "ConvolutionalNeuralNetwork":
                model['name'] = "CNN"
            elif model['name'] == "FeedForwardNeuralNetwork":
                model['name'] = "FFNN"
            elif model['name'] == 'RecurrentNeuralNetworkLSTM':
                model['name'] = "RNN LSTM"
            elif model['name'] == "RecurrentNeuralNetworkGRU":
                model['name'] = "RNN GRU"
            elif model['name'] == "LogisticRegression":
                model['name'] = "Log. Regr."
            if model['neural']:
                neural_models_list.append(model['name'])
                accuracies.append(compute_avg_of_list(model['accuracy']))
                losses.append(compute_avg_of_list(model['loss']))
            acc, prec, rec, f1 = compute_avg_accuracy(model['true'], model['predictions'])
            accuracy_scores.append(acc)
            precision_scores.append(prec)
            recall_scores.append(rec)
            f1_scores.append(f1)
            conf_matrices.append(compute_avg_conf_matrix(model['true'], model['predictions']))
            auc.append(compute_avg_roc_curves(model['name'], model['true'], model['predictions']))
            all_models_list.append(model['name'])

    epochs = [len(acc) for acc in accuracies]

    """
    TEST FOR EMBEDDINGS
    """
    words = ["Word1", "Word2", "Word3", "Word4"]
    embeddings = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    dimensions = 3

    return prepare_dashboard(neural_models_list=neural_models_list, 
                        accuracies=accuracies, 
                        losses=losses, 
                        epochs=epochs, 
                        all_models=all_models_list,
                        accuracy_scores=accuracy_scores,
                        precision_scores=precision_scores,
                        recall_scores=recall_scores,
                        f1_scores=f1_scores,
                        confusion_matrices=conf_matrices,
                        auc=auc,
                        words=words,
                        embeddings=embeddings,
                        dimensions=dimensions)

def get_embed(filename):
    """
    TEST FOR EMBEDDINGS
    """
    embeddings = np.load("data/embed_mat.npy")

    with open("data/int2word.json") as infile:
        for line in infile:
            int2word = json.loads(line)

    embeddings = embeddings[:-1]
    #print(len(embeddings))
    #print(len(int2word.keys()))
    words = []
    for i in range(len(embeddings)):
        try:
            words.append(int2word[str(i)])
        except:
            print("Word is not known.")



    #words = ["Word1", "Word2", "Word3", "Word4"]
    #embeddings = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    dimensions = 3

    layout = html.Div(
                    children=[html.Div(
                    children=
                     html.Div(
                        children=html.Div([
                            dcc.Graph(
                                id='viz-graph',
                                figure=viz_embeddings(embeddings, words, dimensions),
                                style={'width': '100%', 'height':'800px', 'display': 'inline-block'}
                            )
                        ], style={'display': 'inline-block', 'border-width': '0px 0px 0px 0px', 'width':'100%', 'height':'900px', 'margin-top':'10px', 'margin-bottom': '10px'},
                    ))
                    )
                    ]
                )
                
    return layout

#TO FINISH BEFORE FRIDAY
def get_test_performance(filename):
    layout = html.Div(
                children=[html.Div(
                    children=html.Div([
                        dcc.Graph(
                            id='left-top-graph',
                            figure=plot_training_performances(neural_models_list, accuracies, losses, epochs),
                            style={'width': '50%', 'height':'500px', 'display': 'inline-block'}
                        ),
                        dcc.Graph(
                            id='right-top-graph',
                            figure=construct_conf_matrices(confusion_matrices, all_models),
                            style={'width': '50%', 'height':'500px', 'display': 'inline-block'}
                        )
                    ], style={'display': 'inline-block', 'border-width': '0px 0px 0px 0px', 'width':'100%', 'height':'500px'},
                )),
                html.Div(
                    children=html.Div([
                        dcc.Graph(
                            id='middle-graph',
                            figure=performance_table(accuracy_scores, precision_scores, recall_scores, f1_scores, all_models),
                            style={'width': '100%', 'height':'500px', 'display': 'inline-block'}
                        )
                    ], style={'display': 'inline-block', 'border-width': '0px 0px 0px 0px', 'width':'100%', 'height':'600px', 'margin-top':'10px', 'margin-bottom': '10px'},
                ))
                ]
            )
    
    return layout

app = dash.Dash(__name__)
server = app.server
server.secret_key = os.environ.get('SECRET_KEY', 'jfdksgndfjkgdsiugegrhuh239u2ygioajg298jwim')
# CORS(server)

layout_e3 = get_data("data/vis_results_e_3.txt")
layout_e4 = get_data("data/vis_results_e_4.txt")
layout_e5 = get_data("data/vis_results_e_5.txt")
layout_e6 = get_data("data/vis_results_e_6.txt")
layout_wordembedd = get_embed("data/vis_results_e_6.txt")
#layout_test = get_test_performance("data/vis_results_e_6.txt")


app.layout = html.Div([
    html.H1("Second Year Project - Phase 1 results"),
    dcc.Tabs(id='sent_tabs', value='e3_tab', children=[
        dcc.Tab(label='3 Epochs', value='e3_tab'),
        dcc.Tab(label='4 Epochs', value='e4_tab'),
        dcc.Tab(label='5 Epochs', value='e5_tab'),
        dcc.Tab(label='6 Epochs', value='e6_tab'),
        dcc.Tab(label='Word Embeddings', value='word_embedding')#,
        #dcc.Tab(label='Performance on Test', value='perf_on_test')
    ]),
    html.Div(id='sent_content')
])

    
@app.callback(Output('sent_content', 'children'),
            [Input('sent_tabs', 'value')])
def render_content(tab):

    if tab == 'e3_tab':
        return layout_e3
    elif tab == 'e4_tab':
        return layout_e4
    elif tab == 'e5_tab':
        return layout_e5
    elif tab == 'e6_tab':
        return layout_e6
    elif tab == 'word_embedding':
        return layout_wordembedd
    """
    elif tab == 'perf_on_test':
        return layout_test
    """


if __name__ == '__main__':

    launch_dashboard(app)


    