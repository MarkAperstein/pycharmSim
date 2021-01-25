import matplotlib.pyplot as plt

def stochastic_plot(list_of_dfs,title=None,t_labels=None):
    fig,ax=plt.subplots()
    df=list_of_dfs[0]
    keys=df.keys()
    plt.gca().set_prop_cycle(None)
    for i,key in enumerate(keys):
        ax.plot(df.values[:,i],label=key)

    for i in range(1,len(list_of_dfs)):
        plt.gca().set_prop_cycle(None)
        ax.plot(list_of_dfs[i].values)
    ax.legend()
    return fig

def scatter_vs_plot(scatter_df,plot_df,title=None,t_labels=None):
    fig,ax=plt.subplots()
    return fig