import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sys
sys.path.append('../../')

# デバイスの設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_cpu = torch.device("cpu")

# 学習済みモデルをロードする
model = torch.load('./model.pth')
model = model.to(device)

# dataをloadする(float32→tensor→to.device)
tr_data = torch.from_numpy((np.load('./tr_data.npy')).astype(np.float32)).to(device)
tr_label = torch.from_numpy((np.load('./tr_label.npy')).astype(np.float32)).to(device)
va_data = torch.from_numpy((np.load('./va_data.npy')).astype(np.float32)).to(device)
va_label = torch.from_numpy((np.load('./va_label.npy')).astype(np.float32)).to(device)

# outputとlabelを用意する(メモリの関係で6×10枚に絞っている)
tr_label = tr_label[:60].detach().cpu().numpy()
va_label = va_label[:60].detach().cpu().numpy()

tr_output = model(tr_data[:60]).detach().cpu().numpy()
va_output = model(va_data[:60]).detach().cpu().numpy()


# モデルの出力(output)とlabelの比較画像を6×10枚保存する
def mk_image(output, label, mode):

    # 6×10個の結果を確認する
    for i in range(10):
        fig = plt.figure(figsize=(6,2))
        fig.subplots_adjust(hspace=0.2, wspace=0.6)

        # output用
        ax1 = fig.add_subplot(2, 6, 1)
        ax2 = fig.add_subplot(2, 6, 2)
        ax3 = fig.add_subplot(2, 6, 3)
        ax4 = fig.add_subplot(2, 6, 4)
        ax5 = fig.add_subplot(2, 6, 5)
        ax6 = fig.add_subplot(2, 6, 6)

        im1 = ax1.imshow(output[i*6][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6][0].max(),label[i*6][0].max()))
        im2 = ax2.imshow(output[i*6+1][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+1][0].max(),label[i*6+1][0].max()))
        im3 = ax3.imshow(output[i*6+2][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+2][0].max(),label[i*6+2][0].max()))
        im4 = ax4.imshow(output[i*6+3][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+3][0].max(),label[i*6+3][0].max()))
        im5 = ax5.imshow(output[i*6+4][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+4][0].max(),label[i*6+4][0].max()))
        im6 = ax6.imshow(output[i*6+5][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+5][0].max(),label[i*6+5][0].max()))


        divider = make_axes_locatable(ax1)
        cax1 = divider.append_axes('right', size='5%', pad=0.05)
        cbar1 = fig.colorbar(im1, cax1)
        cbar1.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes('right', size='5%', pad=0.05)
        cbar2 = fig.colorbar(im2, cax2)
        cbar2.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes('right', size='5%', pad=0.05)
        cbar3 = fig.colorbar(im3, cax3)
        cbar3.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax4)
        cax4 = divider.append_axes('right', size='5%', pad=0.05)
        cbar4 = fig.colorbar(im4, cax4)
        cbar4.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax5)
        cax5 = divider.append_axes('right', size='5%', pad=0.05)
        cbar5 = fig.colorbar(im5, cax5)
        cbar5.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax6)
        cax6 = divider.append_axes('right', size='5%', pad=0.05)
        cbar6 = fig.colorbar(im6, cax6)
        cbar6.ax.tick_params(labelsize=5)


        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax5.set_xticks([])
        ax5.set_yticks([])
        ax6.set_xticks([])
        ax6.set_yticks([])

        ax1.set_ylabel('output')

        # label用
        ax7 = fig.add_subplot(2, 6, 7)
        ax8 = fig.add_subplot(2, 6, 8)
        ax9 = fig.add_subplot(2, 6, 9)
        ax10 = fig.add_subplot(2, 6, 10)
        ax11 = fig.add_subplot(2, 6, 11)
        ax12 = fig.add_subplot(2, 6, 12)

        im7 = ax7.imshow(label[i*6][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6][0].max(),label[i*6][0].max()))
        im8 = ax8.imshow(label[i*6+1][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+1][0].max(),label[i*6+1][0].max()))
        im9 = ax9.imshow(label[i*6+2][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+2][0].max(),label[i*6+2][0].max()))
        im10 = ax10.imshow(label[i*6+3][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+3][0].max(),label[i*6+3][0].max()))
        im11 = ax11.imshow(label[i*6+4][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+4][0].max(),label[i*6+4][0].max()))
        im12 = ax12.imshow(label[i*6+5][0], cmap=cm.jet, vmin=0, vmax=max(output[i*6+5][0].max(),label[i*6+5][0].max()))


        divider = make_axes_locatable(ax7)
        cax7 = divider.append_axes('right', size='5%', pad=0.05)
        cbar7 = fig.colorbar(im7, cax7)
        cbar7.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax8)
        cax8 = divider.append_axes('right', size='5%', pad=0.05)
        cbar8 = fig.colorbar(im8, cax8)
        cbar8.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax9)
        cax9 = divider.append_axes('right', size='5%', pad=0.05)
        cbar9 = fig.colorbar(im9, cax9)
        cbar9.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax10)
        cax10 = divider.append_axes('right', size='5%', pad=0.05)
        cbar10 = fig.colorbar(im10, cax10)
        cbar10.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax11)
        cax11 = divider.append_axes('right', size='5%', pad=0.05)
        cbar11 = fig.colorbar(im11, cax11)
        cbar11.ax.tick_params(labelsize=5)

        divider = make_axes_locatable(ax12)
        cax12 = divider.append_axes('right', size='5%', pad=0.05)
        cbar12 = fig.colorbar(im12, cax12)
        cbar12.ax.tick_params(labelsize=5)


        ax7.set_xticks([])
        ax7.set_yticks([])
        ax8.set_xticks([])
        ax8.set_yticks([])
        ax9.set_xticks([])
        ax9.set_yticks([])
        ax10.set_xticks([])
        ax10.set_yticks([])
        ax11.set_xticks([])
        ax11.set_yticks([])
        ax12.set_xticks([])
        ax12.set_yticks([])

        ax7.set_ylabel('label')

        if mode == 'tr':
            fig.savefig('image/tr_table/table_' + str(i+1) + '.png')

        if mode == 'va':
            fig.savefig('image/va_table/table_' + str(i+1) + '.png')


# tr_outputとtr_labelを比較する
mk_image(tr_output,tr_label,mode='tr')

# va_outputとva_labelを比較する
mk_image(va_output,va_label,mode='va')