import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from typing import (
    List, Callable, Optional, Dict
)

class NN4NLPPlots:
    '''
    Dummy class to aggregate helper functions
    '''

    @staticmethod
    def plot_embeddings(
                word_embeddings:np.ndarray, 
                vocab:List[str],
                perplexity:Optional[int]=30
            ) -> None:
        # Create TSNE model
        tsne = TSNE(n_components=2, random_state=0, perplexity=perplexity)
        print(word_embeddings.shape)
        word_embeddings_2d = tsne.fit_transform(word_embeddings)

        # Plotting the results with labels from vocab
        plt.figure(figsize=(15, 15))
        for i, word in enumerate(vocab):  
            plt.scatter(word_embeddings_2d[i, 0], word_embeddings_2d[i, 1])
            plt.annotate(word, (word_embeddings_2d[i, 0], word_embeddings_2d[i, 1]))

        plt.xlabel("t-SNE component 1")
        plt.ylabel("t-SNE component 2")
        plt.title("Word Embeddings visualized with t-SNE")
        plt.show()

    @staticmethod
    def plot(COST,ACC):
        fig, ax1 = plt.subplots()
        color = 'tab:red'
        ax1.plot(COST, color=color)
        ax1.set_xlabel('epoch', color=color)
        ax1.set_ylabel('total loss', color=color)
        ax1.tick_params(axis='y', color=color)
        
        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('accuracy', color=color)  # you already handled the x-label with ax1
        ax2.plot(ACC, color=color)
        ax2.tick_params(axis='y', color=color)
        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        
        plt.show()
        plt.close()

    @staticmethod
    def plot_frequencies(
                fdist: Dict[str, int], 
                max_words: Optional[int]=10,
                reversed: Optional[bool]=True
            ) -> None:
        most_freq = list(fdist.items())
        most_freq = np.array(sorted(most_freq, key=lambda x: x[1], reverse=reversed))[:max_words,:]
        plt.figure(figsize=(0.4 * max_words,3))
        plt.bar(most_freq[:,0], most_freq[:,1].astype(int))
        plt.xlabel("Palabra")
        plt.ylabel("Frecuencia")
        plt.xticks(rotation=45)
        plt.show()
        plt.close()

    @staticmethod
    def plot_loss_acc_f1(train_losses, eval_losses, accurracies, f1s) -> None:
        num_epochs = len(train_losses)
        fig,axes = plt.subplots(
            1,3,
            figsize=(9,3),
            tight_layout=True
        )
        sns.lineplot(ax = axes[0], x=range(1, num_epochs + 1), y=train_losses, label='Training Loss')
        sns.lineplot(ax = axes[0], x=range(1, num_epochs + 1), y=eval_losses, label='Evaluation Loss')
        sns.lineplot(ax = axes[1], x=range(1, num_epochs + 1), y=accurracies, label='Accurracy')
        sns.lineplot(ax = axes[2], x=range(1, num_epochs + 1), y=f1s, label='F1 score')

        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accurracy')
        axes[2].set_ylabel('F1 score')
        axes[0].set_title('Training and Evaluation Loss')
        axes[1].set_title('Accurracy')
        axes[2].set_title('F1 score')
        plt.legend()
        plt.show()