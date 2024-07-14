import matplotlib.pyplot as plt
import numpy as np
import json


MODEL = 'MultiVAE'
STEP = 6
MAXIT = 20

def make_plot(x, y, title, ylab = '', vl = 0):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, color = 'blue', linestyle = 'dashed')
    ax.set_xticks(range(len(x)))

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    if vl > 0:
        ax.set_xlabel('Epochs', size = 20)
        ax.vlines(np.arange(len(x), step=vl), ymin=min(y), ymax= max(y), colors='red',linestyles='dotted')
    
    else:
        ax.set_xlabel('Training Step', size = 20)

    ax.set_ylabel(ylab, size = 20)
    ax.tick_params(axis='both', which='major', labelsize=20)
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
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = 'ROG[km]'

                make_plot(x, y, title, ylab, vl)

            case 'rog_ind_2':
                # Diversity of Items
                title = '2k Radius Of Gyration'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = '2-ROG[km]'

                make_plot(x, y, title, ylab, vl)

            case 'D_ind':
                # Diversity of Items
                title = 'Distinct Location Visited'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = 'Nr. of Distinct Location Visited by each user'

                make_plot(x, y, title, ylab, vl)

            case 'L_old_ind':
                # Diversity of Items
                title = 'Old Location Suggested'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = 'Nr. of Distinct Old Location '

                make_plot(x, y, title, ylab, vl)

            case 'L_new_ind':
                # Diversity of Items
                title = 'New Location Suggested'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = 'Nr. of Distinct new Location '

                make_plot(x, y, title, ylab, vl)

            case 'S_ind':
                # Diversity of Items
                title = 'Individual Entropy'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = ''

                make_plot(x, y, title, ylab, vl)

            case 'S_col':
                # Diversity of Items
                title = 'Collective Entropy'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = ''

                make_plot(x, y, title, ylab, vl)

            case 'Expl_ind':
                # Diversity of Items
                title = 'Exploring Events'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = ' '

                make_plot(x, y, title, ylab, vl)

            case 'Ret_ind':
                # Diversity of Items
                title = 'Returning Events'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP

                make_plot(x, y, title, ylab, vl)

            case 'Gini_ind':
                # Diversity of Items
                title = 'Individual Gini'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = STEP
                ylab = ''

                make_plot(x, y, title, ylab, vl)

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

            case _:
                print('metric not implemented')

                
