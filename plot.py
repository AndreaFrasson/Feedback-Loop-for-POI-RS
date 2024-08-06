import matplotlib.pyplot as plt
import numpy as np
import json


MODEL = 'MultiVAE_cp'
STEP = 10
MAXIT = 10
P = 0.8


def make_scatter(array, title, ylab = ''):

    means = np.mean(array, axis = 0)
    errors = np.std(array, axis = 0)
    x = range(MAXIT)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.errorbar(x, means, errors, marker='o', color = 'black', linewidth = 3, elinewidth = 1, capsize=5)
    ax.set_xticks(x)

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax.set_xlabel('Epochs', size = 20)

    ax.set_ylabel(ylab, size = 20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    bo,up = ax.get_ylim()
    try:
        bo -= np.absolute(np.log10(bo))
    except:
        bo = bo
    ax.set_ylim(max(0, bo), up +np.absolute(np.log10(up)))

    plt.grid(color = 'grey', alpha = 0.2, linewidth = 2)  #just add this

    #ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('plot/'+ title +'_'+MODEL+'_'+str(STEP)+'-'+str(MAXIT)+'_'+str(P)+'.png')

    return 

if __name__ == '__main__':

    # reading the data from the file 
    with open('output/'+MODEL+'_'+str(STEP)+'-'+str(MAXIT)+'_'+str(P)+'.txt', 'r') as f:
        data = f.read().replace('\'', '\"')
      
    # reconstructing the data as a dictionary 
    js = json.loads(data) 

    for k in js.keys():

        match k:
            case 'L_col':
                # Diversity of Items
                title = 'Diversity of Items'
                ylab = 'Nr. of Distinct Proposed Location'

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'rog_ind':
                # Diversity of Items
                title = 'Total Radius Of Gyration'
                ylab = 'ROG[km]'

                make_scatter(np.array(js[k]).reshape(MAXIT, -1), title, ylab)

            case 'rog_ind_2':
                # Diversity of Items
                title = 'Total 2k Radius Of Gyration'
                ylab = '2-ROG[km]'

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'D_ind':
                # Diversity of Items
                title = 'Distinct Location Visited'
                ylab = 'Nr. of Distinct Location Visited by each user'

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'L_old_ind':
                # Diversity of Items
                title = 'Old Location Suggested'
                ylab = 'Nr. of Distinct Old Location '

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'L_new_ind':
                # Diversity of Items
                title = 'New Location Suggested'
                ylab = 'Nr. of Distinct new Location '

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'S_ind':
                # Diversity of Items
                title = 'Individual Entropy'
                ylab = ''

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'S_col':
                # Diversity of Items
                title = 'Collective Entropy'
                ylab = ''

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'Expl_ind':
                # Diversity of Items
                title = 'Exploring Events'
                ylab = ' '

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'Ret_ind':
                # Diversity of Items
                title = 'Returning Events'

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'Gini_ind':
                # Diversity of Items
                title = 'Individual Gini'
                ylab = ''

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'test_hit':
               # hit rate
                title = 'Hit Rate'
                ylab = ''

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'test_precision':
                # hit rate
                title = 'Precision'
                ylab = ''

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case 'test_rec':
                # Diversity of Items
                title = 'Recall'
                ylab = ''

                make_scatter(np.array(js[k]).reshape(-1, MAXIT), title, ylab)

            case _:
                print('metric not implemented')