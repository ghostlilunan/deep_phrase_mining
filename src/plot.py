import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pickle


def nll_loss_plot(points, file_name, img_path, colors):
    plt.figure()
    fig, ax = plt.subplots()
    img_path += '{}.png'.format(file_name)
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)

    for i, point in enumerate(points):
        plt.plot(point, colors[i], label="{}-gram".format(i+1))

    legend = ax.legend(loc='center right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize('large')

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width

    plt.savefig(img_path)


def n_gram_nll_loss_plot(file):
    with open(file, 'rb') as f:
         data = pickle.load(f)

    epoch_losses = [data['epoch_losses']]
    colors = ['b']
    # nll_loss_plot(epoch_losses, "epoch_losses", '../local_result/', colors)
    bin_size = [23, 28, 9, 3, 1]
    batch_losses = dict(data['batch_losses'])
    values = list(batch_losses.values())
    results = []

    for i, value in enumerate(values):
        # print(i)
        # print(value)
        results.append([])
        for j in range(0, len(value), bin_size[i]):
            sum = 0
            for k in value[j:j+bin_size[i]]:
                sum += k
            results[i].append(sum/bin_size[i])
        print(len(results[i]))

    print(type(values[0]))
    colors = ['r', 'b', 'g', 'c', 'm']
    nll_loss_plot(results, "batch_losses", '../local_result/', colors)


def bar_chart(x,  y, filename):
    plt.bar(x, y, align='center', alpha=0.5)
    plt.savefig(filename+'.png')


if __name__ == '__main__':
    # n_gram_nll_loss_plot('../data/losses.pkl')
    bar_chart([1, 2, 3, 4, 5],[10, 10, 10, 10, 10])
