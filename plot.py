import matplotlib.pyplot as plt
import numpy as np
import json


MODEL = 'Pop_ir'
STEP = 10
MAXIT = 10

def make_plot(x, y, title, ylab = '', vl = 0, x2 = None):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, color = 'blue', linestyle = 'dashed')
    ax.set_xticks(range(len(x)))

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    ax.set_xlabel('Epochs', size = 20)

    ax.set_ylabel(ylab, size = 20)
    ax.tick_params(axis='both', which='major', labelsize=20)

    #ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('plot/'+ title +'_'+MODEL+'_'+str(STEP)+'-'+str(MAXIT)+'.png')

    return 

def make_scatter(x, mean, var, title, ylab = '', vl = 0, x2 = None):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.errorbar(x, mean, var, linestyle='None', marker='o')
    ax.set_xticks(range(len(x)))

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
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('plot/'+ title +'_'+MODEL+'_'+str(STEP)+'-'+str(MAXIT)+'.png')

    return 

if __name__ == '__main__':

    # reading the data from the file 
    with open('output/'+MODEL+'_'+str(STEP)+'-'+str(MAXIT)+'.txt', 'r') as f:
        data = f.read().replace('\'', '\"')
      
    # reconstructing the data as a dictionary 
    js = json.loads(data) 

    for k in js.keys():

        match k:
            case 'L_col':
                # Diversity of Items
                title = 'Diversity of Items'
                ylab = 'Nr. of Distinct Proposed Location'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP

                make_plot(x, y, title, ylab, vl)

            case 'rog_ind':
                # Diversity of Items
                title = 'Total Radius Of Gyration'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = 'ROG[km]'

                make_scatter(x, means, var, title, ylab, vl)

            case 'rog_ind_2':
                # Diversity of Items
                title = 'Total 2k Radius Of Gyration'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = '2-ROG[km]'

                make_scatter(x, means, var, title, ylab, vl)

            case 'D_ind':
                # Diversity of Items
                title = 'Distinct Location Visited'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = 'Nr. of Distinct Location Visited by each user'

                make_scatter(x, means, var, title, ylab, vl)

            case 'L_old_ind':
                # Diversity of Items
                title = 'Old Location Suggested'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = 'Nr. of Distinct Old Location '

                make_scatter(x, means, var, title, ylab, vl)

            case 'L_new_ind':
                # Diversity of Items
                title = 'New Location Suggested'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = 'Nr. of Distinct new Location '

                make_scatter(x, means, var, title, ylab, vl)

            case 'S_ind':
                # Diversity of Items
                title = 'Individual Entropy'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = ''

                make_scatter(x, means, var, title, ylab, vl)

            case 'S_col':
                # Diversity of Items
                title = 'Collective Entropy'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = ''

                make_scatter(x, means, var, title, ylab, vl)

            case 'Expl_ind':
                # Diversity of Items
                title = 'Exploring Events'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = ' '

                make_scatter(x, means, var, title, ylab, vl)

            case 'Ret_ind':
                # Diversity of Items
                title = 'Returning Events'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP

                make_scatter(x, means, var, title, ylab, vl)

            case 'Gini_ind':
                # Diversity of Items
                title = 'Individual Gini'
                means = [i[0] for i in js[k]]
                var = [i[1] for i in js[k]]
                x = [i for i in range(len(js[k]))]
                vl = STEP
                ylab = ''

                make_scatter(x, means, var, title, ylab, vl)

            case 'test_hit':
               # hit rate
                title = 'Hit Rate'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                ylab = ''

                make_plot(x, y, title, ylab)

            case 'test_precision':
                # hit rate
                title = 'Precision'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                ylab = ''

                make_plot(x, y, title, ylab)

            case 'test_rec':
                # Diversity of Items
                title = 'Recall'
                ylab = ''
                x = [i for i in range(len(js[k]))]
                y = js[k]

                make_plot(x, y, title, ylab)

            case _:
                print('metric not implemented')