# plotly and matplotlib libraries
import plotly.plotly as py
import plotly.offline
import plotly.graph_objs as go
import plotly.io as pio
import plotly.figure_factory as ff
from plotly.offline import plot_mpl
from plotly import tools

# sklearn libraries
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# other libraries
import numpy as np
import time
import os
from shutil import copyfile
from itertools import cycle
from scipy import interp


import warnings
warnings.filterwarnings("ignore")


def plot3D(embeddings, text, dimensions):

    embeddings_reduced = PCA(n_components=dimensions).fit_transform(embeddings)

    x = embeddings_reduced[: ,0]
    y = embeddings_reduced[: ,1]
    z = embeddings_reduced[: ,2]

    trace = go.Scatter3d(x=x, y=y, z=z, text=text, mode='markers',
                        marker=dict(
                                    size=4,
                                    line=dict(
                                            color='rgba(217, 217, 217, 0.14)',
                                            width=0.3
                                    ),
                                    opacity=0.8
                        )
    )

    fig = save_plot(trace, dimensions)
    return fig


def plot2D(embeddings, text, dimensions):

    embeddings_reduced = PCA(n_components=dimensions).fit_transform(embeddings)

    x = embeddings_reduced[: ,0]
    y = embeddings_reduced[: ,1]

    trace = go.Scatter(x=x, y=y, text=text, mode='markers',
                        marker=dict(
                                    size=8,
                                    line=dict(
                                            color='rgba(217, 217, 217, 0.14)',
                                            width=0.5
                                    ),
                                    opacity=0.8
                        )
    )
    
    fig = save_plot(trace, dimensions)
    return fig


def save_plot(trace, dimensions):
    data = [trace]
    fig = go.Figure(data=data)
    return fig
    #pio.write_image(fig, f'viz/{model}_{dimensions}d-embeddings.svg')
    #plotly.offline.plot(fig, filename=f'viz/{model}_{dimensions}d-embeddings.html', auto_open=False)


def viz_embeddings(embeddings, text, dimensions=3):
    """
    Args:
        embeddings (ndarray): numpy array containing the embeddings
        text (list): list containing the word corresponding to the embedding 
        dimensions (int, optional): Defaults to 3. 
                                Number of dimension in which we want to display the words
    """

    assert isinstance(dimensions, int)
    assert len(embeddings) == len(text)

    if dimensions < 2 or dimensions > 3:
        print(f"Impossible to display {dimensions} dimensions.")
        return

    if dimensions == 2:
        fig = plot2D(embeddings, text, dimensions)
    if dimensions == 3:
        fig = plot3D(embeddings, text, dimensions)
    
    fig['layout'].update(title="Word Embeddings")

    return fig


def plot_confusion_matrix(model, confusion_matrix, xre, yre):

    #assert set(pred_labels).issubset(set(true_labels))
    label_set = ['pos','neg']
    values = confusion_matrix

    # colorscale stolen from https://plot.ly/python/colorscales/#custom-heatmap-colorscale

    """
    'Greys', 'YlGnBu', 'Greens', 'YlOrRd', 'Bluered', 'RdBu',
            'Reds', 'Blues', 'Picnic', 'Rainbow', 'Portland', 'Jet',
            'Hot', 'Blackbody', 'Earth', 'Electric', 'Viridis', 'Cividis'
    """
    colorscale = [
        # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
        [0, '#beeff9'],
        [0.1, '#beeff9'],

        # Let values between 10-20% of the min and max of z
        # have color rgb(20, 20, 20)
        [0.1, '#add4f6'],
        [0.2, '#add4f6'],

        # Values between 20-30% of the min and max of z
        # have color rgb(40, 40, 40)
        [0.2, '#9cb9f4'],
        [0.3, '#9cb9f4'],

        [0.3, '#8b9ff2'],
        [0.4, '#8b9ff2'],

        [0.4, '#7a84f0'],
        [0.5, '#7a84f0'],

        [0.5, '#696aed'],
        [0.6, '#696aed'],

        [0.6, '#584feb'],
        [0.7, '#584feb'],

        [0.7, '#4735e9'],
        [0.8, '#4735e9'],

        [0.8, '#361ae7'],
        [0.9, '#361ae7'],

        [0.9, '#2500e5'],
        [1.0, '#2500e5']
    ]
    font_colors = ['#efecee', '#3c3636']

    trace = {
        "x": label_set, 
        "y": label_set[::-1], 
        "z": values[::-1], 
        "type": "heatmap",
        #"colorscale": colorscale,
        #"fontcolor": font_colors,
        "autocolorscale": True,
        "showscale": False,
        "zmin": 0,
        "zmax": 36500
    }

    annotations = []
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            d = {}
            d["x"] = str(label_set[row])
            d["y"] = str(label_set[column])
            d["font"] = {"color": "white"}
            d["showarrow"] = False
            d["text"]= str(values[column][row])
            d["xref"] = f"x{xre}"
            d["yref"] = f"y{yre}"
            annotations.append(d)

    return trace, annotations
  

def construct_conf_matrices(confusion_matrices, all_models):
    rows = 2
    cols = 3
    gen_fig = tools.make_subplots(rows=rows, cols=cols)

    traces = []
    annotations = []
    for i in range(len(all_models)):
        trace, annot = plot_confusion_matrix(all_models[i], confusion_matrices[i], i+1, i+1)
        traces.append(trace)
        annotations = annotations + annot
        
    gen_fig.append_trace(traces[0], 1, 1)
    gen_fig.append_trace(traces[1], 1, 2)
    gen_fig.append_trace(traces[2], 1, 3)
    gen_fig.append_trace(traces[3], 2, 1)
    gen_fig.append_trace(traces[4], 2, 2)
    gen_fig.append_trace(traces[5], 2, 3)
    
    gen_fig['layout'].update(title='Confusion Matrices')
    for i in range(len(all_models)):
        gen_fig['layout'][f'xaxis{i+1}'].update(title=all_models[i])
        if i in [0,3]:
            gen_fig['layout'][f'yaxis{i+1}'].update(title='True class')

    gen_fig['layout'].update(annotations=annotations)
    
    gen_fig['layout']['yaxis'].update(domain=[0.67, 1.0])
    gen_fig['layout']['yaxis2'].update(domain=[0.67, 1.0])
    gen_fig['layout']['yaxis3'].update(domain=[0.67, 1.0])
    gen_fig['layout']['yaxis4'].update(domain=[0.0, 0.4])
    gen_fig['layout']['yaxis5'].update(domain=[0.0, 0.4])
    gen_fig['layout']['yaxis6'].update(domain=[0.0, 0.4])

    #plotly.offline.plot(gen_fig, filename='viz/confusion_matrices.html', auto_open=True)
    #pio.write_image(gen_fig, 'viz/confusion_matrices.svg')
    return gen_fig


def plot_training_performances(models, accuracies, losses, epochs=[15,15,15,15]):

    assert len(models) == len(epochs)

    x_data = []
    for epoch in range(len(epochs)):
        assert len(accuracies[epoch]) == epochs[epoch]
        assert len(losses[epoch]) == epochs[epoch]
        x_data.append([i+1 for i in range(epochs[epoch])])

    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Accuracies', 'Losses'))

    colors = ['#173f5f', '#20639b', '#3caea3', '#f6d55c', '#ed553b']
    for i in range(0, len(models)):
        fig.append_trace(go.Scatter(
            x=x_data[i],
            y=accuracies[i],
            mode='lines',
            legendgroup=f'group_{i}',
            name=models[i],
            line=dict(color=colors[i], width=2),
            connectgaps=True,
            showlegend=True
        ), 1, 1)

        fig.append_trace(go.Scatter(
            x=[x_data[i][0], x_data[i][-1]],
            y=[accuracies[i][0], accuracies[i][-1]],
            mode='markers',
            legendgroup=f'group_{i}',
            name=models[i],
            marker=dict(color=colors[i], size=8),
            showlegend=False
        ), 1, 1)

        fig.append_trace(go.Scatter(
            x=x_data[i],
            y=losses[i],
            mode='lines',
            legendgroup=f'group_{i}',
            name=models[i],
            line=dict(color=colors[i], width=2),
            connectgaps=True,
            showlegend=False
        ), 1, 2)

        fig.append_trace(go.Scatter(
            x=[x_data[i][0], x_data[i][-1]],
            y=[losses[i][0], losses[i][-1]],
            mode='markers',
            legendgroup=f'group_{i}',
            name=models[i],
            marker=dict(color=colors[i], size=8),
            showlegend=False
        ), 1, 2)
        

    annotations = []

    # adding labels
    for y_trace, label, color in zip(accuracies, models, colors):
        # labeling the left_side of the plot
        annotations.append(dict(xref='paper', x=0.00, y=y_trace[0],
                                    xanchor='right', yanchor='middle',
                                    text='{}'.format(y_trace[0]),
                                    font=dict(family='Arial',
                                                size=16),
                                    showarrow=False))
        # labeling the right_side of the plot
        annotations.append(dict(xref='paper', x=0.48, y=y_trace[-1],
                                    xanchor='left', yanchor='middle',
                                    text='{}'.format(y_trace[-1]),
                                    font=dict(family='Arial',
                                                size=16),
                                    showarrow=False))
    # adding title
    annotations.append(dict(xref='paper', yref='paper', x=0.5, y=1.05,
                                xanchor='left', yanchor='bottom',
                                text='Accuracy and Losses',
                                font=dict(family='Arial',
                                            size=30,
                                            color='rgb(37,37,37)'),
                                showarrow=False))


    fig['layout'].update(title='Accuracies and Losses')
    fig['layout']['xaxis1'].update(title='Epoch')
    fig['layout']['xaxis2'].update(title='Epoch')

    fig['layout']['yaxis1'].update(title='Accuracy')
    fig['layout']['yaxis2'].update(title='Loss')

    #pio.write_image(fig, 'viz/train_perf.svg')
    return fig
    #plotly.offline.plot(fig, filename='viz/train_perf.html', auto_open=False)
    

def performance_table(accuracies, precisions, recalls, f1_scores, list_of_models):

    header = [['Model', 'Accuracy', 'Precision', 'Recall','F1-score']]

    baselines = [['Bernoulli <br>Baseline', 0.50, 0.50, 0.50, 0.50],
                ['Human <br>Baseline', 0.50, 0.50, 0.50, 0.50]]

    table_data = []
    for i in range(len(list_of_models)):
        line = []
        line.append(list_of_models[i])
        line.append(round(accuracies[i],2))
        line.append(round(precisions[i],2))
        line.append(round(recalls[i],2))
        line.append(round(f1_scores[i],2))
        table_data.append(line)

    table_data = header + baselines + table_data
    figure = ff.create_table(table_data, height_constant=60)
    figure.layout.margin.update({'t':50, 'b':10})
    figure.layout.update({'title': 'Model Performance Comparison'})
    #pio.write_image(figure, 'viz/comparison_table.svg')
    return figure
    #plotly.offline.plot(figure, filename='viz/comparison_table.html', auto_open=False)

"""NOT USED NOW"""
def compute_roc(list_of_models, y_scores, y_test):
    all_data = []
    auc_scores = []

    for mod in range(len(y_scores)):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(len(y_test[0])):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_scores[mod][:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_scores[mod].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(y_test[0]))]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(y_test[0])):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(y_test[0])

        # Compute macro-average ROC curve and ROC area
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot all ROC curves
        data = []
        trace1 = {
            "x": fpr["micro"],
            "y": tpr["micro"],
            "type": "scatter",
            "mode": "lines",
            "legendgroup": "i",
            "line": dict(color='deeppink', width=2, dash='dot'),
            "name":f'micro-average {list_of_models[mod]} (area = {round(roc_auc["micro"],2)})'
        }
        data.append(trace1)

        trace2 = {
            "x": fpr["macro"],
            "y": tpr["macro"],
            "type": "scatter",
            "mode": "lines",
            "legendgroup": "j",
            "line": dict(color='navy', width=2, dash='dot'),
            "name":f'macro-average {list_of_models[mod]} (area = {round(roc_auc["macro"],2)})'
        }
        data.append(trace2)

        classes=['pos','neg']
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(range(len(y_test[0])), colors):
            trace3 = {
                "x": fpr[i],
                "y": tpr[i],
                "type": "scatter",
                "mode": "lines",
                "legendgroup": f"k{i}",
                "line": dict(color=color, width=2),
                "name":f'{list_of_models[mod]} for class {classes[i]} (area = {round(roc_auc[i],2)})'
            }
            data.append(trace3)

        trace4 = {
                "x": [0, 1],
                "y": [0, 1],
                "mode": "lines",
                "type": "scatter",
                "legendgroup": "l",
                "line": dict(color='black', width=2, dash='dash'),
                "showlegend": False
            }
        data.append(trace4)

        s = 0
        for i in range(len(y_test[0])):
            s += roc_auc[i]
        mean_auc = s / len(y_test[0])

        auc_scores.append(mean_auc)
        all_data.append(data)


    fig = plot_roc_curves(np.array(all_data), auc_scores, list_of_models)
    return fig

"""NOT USED NOW"""
def plot_roc_curves(all_data, auc_scores, list_of_models):
    
    rows = 2
    cols = 3
    gen_fig = tools.make_subplots(rows=rows, cols=cols)

    # yes, it's ugly.
    gen_fig.append_trace(all_data[0][0], 1, 1)
    gen_fig.append_trace(all_data[0][1], 1, 1)
    gen_fig.append_trace(all_data[0][2], 1, 1)
    gen_fig.append_trace(all_data[0][3], 1, 1)
    gen_fig.append_trace(all_data[0][4], 1, 1)
    gen_fig.append_trace(all_data[1][0], 1, 2)
    gen_fig.append_trace(all_data[1][1], 1, 2)
    gen_fig.append_trace(all_data[1][2], 1, 2)
    gen_fig.append_trace(all_data[1][3], 1, 2)    
    gen_fig.append_trace(all_data[1][4], 1, 2)
    gen_fig.append_trace(all_data[2][0], 1, 3)
    gen_fig.append_trace(all_data[2][1], 1, 3)
    gen_fig.append_trace(all_data[2][2], 1, 3)
    gen_fig.append_trace(all_data[2][3], 1, 3)    
    gen_fig.append_trace(all_data[2][4], 1, 3)
    gen_fig.append_trace(all_data[3][0], 2, 1)
    gen_fig.append_trace(all_data[3][1], 2, 1)
    gen_fig.append_trace(all_data[3][2], 2, 1)
    gen_fig.append_trace(all_data[3][3], 2, 1)    
    gen_fig.append_trace(all_data[3][4], 2, 1)
    gen_fig.append_trace(all_data[4][0], 2, 2)
    gen_fig.append_trace(all_data[4][1], 2, 2)
    gen_fig.append_trace(all_data[4][2], 2, 2)
    gen_fig.append_trace(all_data[4][3], 2, 2)    
    gen_fig.append_trace(all_data[4][4], 2, 2)
    gen_fig.append_trace(all_data[5][0], 2, 3)
    gen_fig.append_trace(all_data[5][1], 2, 3)
    gen_fig.append_trace(all_data[1][2], 2, 3)
    gen_fig.append_trace(all_data[5][3], 2, 3)    
    gen_fig.append_trace(all_data[5][4], 2, 3)
    
    gen_fig['layout'].update(title='ROC Curves')
    for i in range(len(list_of_models)):
        gen_fig['layout'][f'xaxis{i+1}'].update(title=f"{list_of_models[i]} <br> AUC: {round(auc_scores[i],2)}")

    gen_fig['layout']['yaxis'].update(domain=[0.67, 1.0])
    gen_fig['layout']['yaxis2'].update(domain=[0.67, 1.0])
    gen_fig['layout']['yaxis3'].update(domain=[0.67, 1.0])
    gen_fig['layout']['yaxis4'].update(domain=[0.0, 0.4])
    gen_fig['layout']['yaxis5'].update(domain=[0.0, 0.4])
    gen_fig['layout']['yaxis6'].update(domain=[0.0, 0.4])
        
    #pio.write_image(gen_fig, 'viz/roc_curves.svg')
    return gen_fig
    #plotly.offline.plot(gen_fig, filename='viz/roc_curve.html', auto_open=True)


def pred_to_labels(preds):

    preds_copy = preds.copy()
    for model_pred in range(len(preds_copy)):
        preds_copy[model_pred] = np.argmax(preds_copy[model_pred], axis=1)
    
    new_pred = []
    for model_labels in range(len(preds_copy)):
        p = []
        for label in preds_copy[model_labels]:
            if label == 1:
                p.append('pos')
            else:
                p.append('neg')
        new_pred.append(p)

    return new_pred

def compute_avg_of_list(list_of_accuracies):
    return np.mean(list_of_accuracies, axis=0).tolist()

def compute_avg_accuracy(true, predictions):
    accuracies, precisions, recalls, f1_scores = [], [], [], []
    for i in range(len(predictions)):
        new_true = []
        new_pred = []
        for j in range(len(true)):
            if true[i][j] == 1:
                new_true.append('pos')
            else:
                new_true.append('neg')
            if predictions[i][j] > .5:
                new_pred.append('pos')
            else:
                new_pred.append('neg')
        
        accuracies.append(accuracy_score(new_true, new_pred))
        precisions.append(precision_score(new_true, new_pred, average='micro'))
        recalls.append(recall_score(new_true, new_pred, average='micro'))
        f1_scores.append(f1_score(new_true, new_pred, average='weighted'))
        
    return np.mean(accuracies), np.mean(precisions), np.mean(recalls), np.mean(f1_scores)

def compute_avg_conf_matrix(true, predictions):
    matrices = []
    label_set = ['pos','neg']
    for i in range(len(predictions)):
        new_true = []
        new_pred = []
        for j in range(len(predictions[i])):
            if true[i][j] == 1:
                new_true.append('pos')
            else:
                new_true.append('neg')
            if predictions[i][j] > .5:
                new_pred.append('pos')
            else:
                new_pred.append('neg')

        matrices.append(confusion_matrix(new_true, new_pred, labels=label_set))

    matrices = np.array(matrices)
    return np.sum(matrices, axis=0)

def compute_avg_roc_curves(model_name, true, predictions):

    all_new_true = []
    all_new_pred = []
    for i in range(len(predictions)):
        new_true = []
        new_pred = []
        for j in range(len(predictions[i])):
            if true[i][j] == 1.0:
                new_true.append([0,1])
            else:
                new_true.append([1,0])
            new_pred.append([1-predictions[i][j], predictions[i][j]])

        new_true = np.array(new_true)
        new_pred = np.array(new_pred)
        all_new_true.append(new_true)
        all_new_pred.append(new_pred)

    all_new_true = np.array(all_new_true)
    all_new_pred = np.array(all_new_pred)
    list_fpr = []
    list_tpr = []
    list_roc_auc = []
    list_all_fpr = []
    list_mean_tpr = []
    for fold in range(len(all_new_pred)):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(2):
            fpr[i], tpr[i], _ = roc_curve(all_new_true[fold][:, i], all_new_pred[fold][:, i], pos_label=1)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_new_true[fold].ravel(), all_new_pred[fold].ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(2)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(2):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= 2

        # Compute macro-average ROC curve and ROC area
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        list_fpr.append(fpr)
        list_tpr.append(tpr)
        list_mean_tpr.append(mean_tpr)
        list_roc_auc.append(roc_auc)
        list_all_fpr.append(all_fpr)

    neg, pos = [], []
    for i in range(len(list_roc_auc)):
        neg.append(list_roc_auc[i][0])
        pos.append(list_roc_auc[i][1])
    
    return [model_name, [np.mean(neg), np.mean(pos)]]
    
def auc_table(auc):
    
    header = [['Model', 'AUC Negative', 'AUC Positive', 'AUC']]

    table_data = []
    for i in range(len(auc)):
        line = []
        line.append(auc[i][0])
        line.append(round(auc[i][1][0],2))
        line.append(round(auc[i][1][1],2))
        line.append(round(np.mean(auc[i][1]),2))
        table_data.append(line)

    table_data = header + table_data
    figure = ff.create_table(table_data, height_constant=60)
    figure.layout.margin.update({'t':50, 'b':10})
    figure.layout.update({'title': 'Areas Under the Curves'})
    #pio.write_image(figure, 'viz/comparison_table.svg')
    return figure
    #plotly.offline.plot(figure, filename='viz/comparison_table.html', auto_open=False)

    
if __name__ == "__main__":

    """
    TEST FOR EMBEDDINGS
    """
    text = ["Word1", "Word2", "Word3", "Word4"]
    #viz_embeddings(np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]), text, 3)

    """
    TEST FOR CONFUSION MATRICES
    """
    NUM_OF_MODELS = 6
    list_of_models = ["SVM", "Logistic Reg.", "Naive Bayes", "FFNN", "CNN", "RNN"]
    true_labels = ["pos", "neg", "neg", "pos", "neg"]
    models_predictions = [[] for _ in range(NUM_OF_MODELS)]
    
    models_predictions[0] = [[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]
    models_predictions[1] = [[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]
    models_predictions[2] = [[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]
    models_predictions[3] = [[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]
    models_predictions[4] = [[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]
    models_predictions[5] = [[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]]

    models_predictions = pred_to_labels(models_predictions)
    #conf_matrices = construct_conf_matrices(true_labels, models_predictions, list_of_models)

    """
    TEST FOR TRAINING PERFORMANCES
    """
    models = ['Log. Regr.', 'FFNN', 'CNN', 'RNN']
    acc = [
        [74, 82, 80, 74, 73, 72, 74, 70, 70, 66, 66, 69],
        [45, 42, 50, 46, 36, 36, 34, 35, 32, 31, 31, 28],
        [13, 14, 20, 24, 20, 24, 24, 40, 35, 41, 43, 50],
        [18, 21, 18, 21, 16, 14, 13, 18, 17, 16, 19, 23],
    ]

    loss = [
        [74, 82, 80, 74, 73, 72, 74, 70, 70, 66, 66, 69],
        [45, 42, 50, 46, 36, 36, 34, 35, 32, 31, 31, 28],
        [13, 14, 20, 24, 20, 24, 24, 40, 35, 41, 43, 50],
        [18, 21, 18, 21, 16, 14, 13, 18, 17, 16, 19, 23],
    ]

    #plot_training_performances(models, acc, loss, batches=[12, 12, 12, 12])
    
    """
    TEST FOR TABLE OF PERFORMANCE COMPARISON
    """
    #performance_table(input)

    """
    TEST FOR ROC CURVES
    """

    y_scores = [[] for _ in range(NUM_OF_MODELS)]
    y_scores[0] = np.array([[0.3, 0.7], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    y_scores[1] = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    y_scores[2] = np.array([[0.4, 0.6], [0.4, 0.6], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    y_scores[3] = np.array([[0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    y_scores[4] = np.array([[0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.8, 0.2], [0.7, 0.3]])
    y_scores[5] = np.array([[0.3, 0.7], [0.4, 0.6], [0.6, 0.4], [0.8, 0.2], [0.7, 0.3]])
    y_test = np.array([[0, 1], [1, 0], [0, 1], [1, 0], [1, 0]])

    #compute_roc(list_of_models, y_scores, y_test)

