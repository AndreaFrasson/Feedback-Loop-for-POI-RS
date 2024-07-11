import matplotlib.pyplot as plt
import numpy as np
import json


MODEL = 'MultiVAE'

def make_plot(x, y, title, vl = 0):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, color = 'blue', linestyle = 'dashed')
    ax.set_xticks(range(len(x)))

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))

    if vl > 0:
        ax.set_xlabel('Epochs', size = 20)
        ax.vlines(np.arange(len(x), step=vl), ymin=min(y), ymax= max(y), colors='red',linestyles='dotted')
    
    else:
        ax.set_xlabel('Training Step', size = 20)

    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('plot/'+ title +'_'+MODEL+'_5-20.png')

    return 


if __name__ == '__main__':

    # reading the data from the file 
    with open('output/'+MODEL+'_5-20.txt', 'r') as f:
        data = f.read().replace('\'', '\"')
      
    # reconstructing the data as a dictionary 
    js = json.loads(data) 

    for k in js.keys():

        match k:
            case 'L_col':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'rog_ind':
                # Diversity of Items
                title = 'Total Radius Of Gyration'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'rog_ind_2':
                # Diversity of Items
                title = '2k Radius Of Gyration'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'D_ind':
                # Diversity of Items
                title = 'Distinct Location Visited'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'L_old_ind':
                # Diversity of Items
                title = 'Old Location Suggested'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'L_new_ind':
                # Diversity of Items
                title = 'New Location Suggested'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'S_ind':
                # Diversity of Items
                title = 'Individual Entropy'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'S_col':
                # Diversity of Items
                title = 'Collective Entropy'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'Expl_ind':
                # Diversity of Items
                title = 'Exploring Events'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'Ret_ind':
                # Diversity of Items
                title = 'Returning Events'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'Gini_ind':
                # Diversity of Items
                title = 'Individual Gini'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'test_hit':
               # hit rate
                title = 'Hit Rate'
                x = [i for i in range(len(js[k]))]
                y = js[k]

                make_plot(x, y, title)

            case 'test_precision':
                # hit rate
                title = 'Precision'
                x = [i for i in range(len(js[k]))]
                y = js[k]

                make_plot(x, y, title)

            case _:
                print('metric not implemented')

                
