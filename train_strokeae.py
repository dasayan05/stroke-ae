import sys, os
import torch, numpy as np
import torch.utils.tensorboard as tb
from torch.nn.utils.rnn import pad_packed_sequence

# from quickdrawstrokes import QuickDrawStrokes
from quickdraw.quickdraw import QuickDraw
from strokeae import RNNStrokeAE, StrokeMSELoss

def main( args ):
    qds = QuickDraw(args.root, max_sketches_each_cat=200, mode=QuickDraw.STROKE, start_from_zero=True,
        verbose=True,
        seperate_p_tensor=True, shifted_seq_as_supevision=True)
    qds_train, qds_test = qds.split(0.8)
    qdl_train = qds_train.get_dataloader(args.batch_size)
    qdl_test = qds_test.get_dataloader(args.batch_size)

    model = RNNStrokeAE(2, args.hidden, args.layers, 2, bidirectional=args.bidirec, ip_free_decoding=self.ip_free_dec)
    model = model.float()
    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    strokemse = StrokeMSELoss()

    curloss = np.inf

    writer = tb.SummaryWriter(os.path.join('.', 'logs', args.tag))

    count = 0
    for e in range(args.epochs):

        model.eval()
        avg_loss, c = 0., 0
        for i, ((X, _), (Y, P), _) in enumerate(qdl_test):
            (Y, L), (P, _) = pad_packed_sequence(Y, batch_first=True), pad_packed_sequence(P, batch_first=True)

            h_initial = torch.zeros(args.layers * (2 if args.bidirec else 1), X.batch_sizes.max(), args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, Y, P = X.cuda(), Y.cuda(), P.cuda()
                h_initial = h_initial.cuda()
            
            out, p = model(X, h_initial)

            loss = strokemse(out, p, Y, P, L)
            avg_loss = ((avg_loss * c) + loss.item()) / (c + 1)
            c += 1
        print(f'[Testing: -/{e}/{args.epochs}] -> Loss: {avg_loss}')
        writer.add_scalar('test_loss', avg_loss, global_step=count)

        model.train()
        for i, ((X, _), (Y, P), _) in enumerate(qdl_train):
            (Y, L), (P, _) = pad_packed_sequence(Y, batch_first=True), pad_packed_sequence(P, batch_first=True)

            h_initial = torch.zeros(args.layers * (2 if args.bidirec else 1), X.batch_sizes.max(), args.hidden, dtype=torch.float32)
            if torch.cuda.is_available():
                X, Y, P = X.cuda(), Y.cuda(), P.cuda()
                h_initial = h_initial.cuda()
            
            out, p = model(X, h_initial)

            loss = strokemse(out, p, Y, P, L)
            
            if i % args.interval == 0:
                count += 1
                if loss < curloss:
                    torch.save(model.state_dict(), args.modelname + '.pth')
                    curloss = loss
                    saved = True
                else:
                    saved = False

                saved_string = ' (saved)' if saved else ''
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {loss}{saved_string}')
                
                writer.add_scalar('train_loss', loss.item(), global_step=count)

            optim.zero_grad()
            loss.backward()
            optim.step()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--ip_free_dec', action='store_true', help='Decoder inputs are zero')
    parser.add_argument('-b','--batch_size', type=int, required=False, default=32, help='batch size')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('--hidden', type=int, required=False, default=128, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in RNN')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=500, help='no of epochs')
    parser.add_argument('--dropout', type=float, required=False, default=0.5, help='dropout fraction')
    parser.add_argument('--bidirec', action='store_true', help='Want the RNN to be bidirectional?')
    parser.add_argument('-i', '--interval', type=int, required=False, default=100, help='logging interval')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    args = parser.parse_args()

    main( args )