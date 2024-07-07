import matplotlib.pyplot as plt
import numpy as np
import json


def make_plot(x, y, title, vl = 0):

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(x, y, color = 'blue', linestyle = 'dashed')
    ax.set_xticks(range(len(x)))
    if vl > 0:
        ax.vlines(np.arange(len(x), step=vl), ymin=min(y), ymax= max(y), colors='red',linestyles='dotted')
    
    ax.set_xlabel('User Degree', fontsize=30)
    ax.set_ylabel('Frequency', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=24)
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig('plot/'+ title +'_Random_5-20.png')


    return 


if __name__ == '__main__':

    # reading the data from the file 
    with open('output/Random_5-20.txt', 'r') as f:
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
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'rog_ind_2':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'D_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'L_old_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'L_new_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'S_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'S_col':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'Expl_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'Ret_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)

            case 'Gini_ind':
                # Diversity of Items
                title = 'Diversity of Items'
                x = [i for i in range(len(js[k]))]
                y = js[k]
                vl = 5

                make_plot(x, y, title, vl)
                
            case _:
               # hit and precision 
                continue
