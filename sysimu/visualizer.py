import matplotlib.pyplot as plt
import numpy as np

def stochastic_plot(list_of_dfs,title=None,dt=1):
    fig,ax=plt.subplots()
    df=list_of_dfs[0]
    keys=df.keys()
    plt.gca().set_prop_cycle(None)

    t=np.array(range(df.values.shape[0]))*dt


    for i,key in enumerate(keys):
        ax.plot(t,df.values[:,i],label=key)

    for i in range(1,len(list_of_dfs)):
        plt.gca().set_prop_cycle(None)
        ax.plot(t,list_of_dfs[i].values)

    if title:
        ax.title.set_text(title)
    ax.legend()
    return fig

def scatter_vs_plot(scatter_df,plot_df,title=None,t_labels=None):
    fig,ax=plt.subplots()
    return fig