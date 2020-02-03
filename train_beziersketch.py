import matplotlib.pyplot as plt
import torch, os, numpy as np
from torch.distributions import Normal
from torch.utils import tensorboard as tb

from quickdraw.quickdraw import QuickDraw
from bezierae import RNNBezierAE, RNNSketchAE, gmm_loss
from infer_beziersketch import inference, drawsketch, stroke_embed

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'mosquito', 'firetruck', 'owl', 'pig', 'face', 'purse', 'shoe' ]

    qd = QuickDraw(args.root, categories=chosen_classes[:args.n_classes], max_sketches_each_cat=args.max_sketches_each_cat,
        verbose=True, normalize_xy=True, start_from_zero=False, mode=QuickDraw.STROKESET, raw=args.raw, npz=args.npz)
    
    qdtrain, qdtest = qd.split(0.8)
    qdltrain, qdltest = qdtrain.get_dataloader(args.batch_size), qdtest.get_dataloader(args.batch_size)

    # chosen device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Embedder model (pretrained and freezed)
    embedder = RNNBezierAE(2, args.embhidden, args.emblayers, args.bezier_degree, bidirectional=True,
        variational=args.embvariational, stochastic_t=args.stochastic_t)
    embmodel = os.path.join(args.base, args.embmodel)
    if os.path.exists(embmodel):
        embedder.load_state_dict(torch.load(embmodel))
    else:
        raise FileNotFoundError('Embedding model not found')
    h_initial_emb = torch.zeros(args.emblayers * 2, args.batch_size, args.embhidden, dtype=torch.float32)
    c_initial_emb = torch.zeros(args.emblayers * 2, args.batch_size, args.embhidden, dtype=torch.float32)
    if torch.cuda.is_available():
        embedder, h_initial_emb, c_initial_emb = embedder.cuda(), h_initial_emb.cuda(), c_initial_emb.cuda()
    embedder.eval()

    # RNN Sketch model
    n_ratw = args.bezier_degree + 1 - 2
    n_ctrlpt = (args.bezier_degree + 1) * 2
    model = RNNSketchAE((n_ctrlpt, n_ratw, 2), args.hidden, dropout=args.dropout, n_mixture=args.n_mix)
    
    h_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
    c_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
    if torch.cuda.is_available():
        model, h_initial, c_initial = model.cuda(), h_initial.cuda(), c_initial.cuda()

    optim = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))

    count, best_loss = 0, np.inf
    for e in range(args.epochs):
        model.train()
        for i, B in enumerate(qdltrain):
            with torch.no_grad():
                ctrlpts, ratws, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)
            
            if args.rendersketch:
                fig, ax = plt.subplots(5, 5, figsize=(20, 20))
                for b in range(ctrlpts.shape[0]):
                    if b > 4:
                        break
                    drawsketch(ctrlpts[b], ratws[b], starts[b], n_strokes[b], draw_axis=ax[b, 0])
                    ax[b, 0].invert_yaxis()
                    for j in range(1, 5):
                        cdist = Normal(ctrlpts[b], torch.ones_like(ctrlpts[b]) * 0.01)
                        rdist = Normal(ratws[b], torch.ones_like(ratws[b]) * 0.01)
                        sdist = Normal(starts[b], torch.ones_like(starts[b]) * 0.01)
                        drawsketch(cdist.sample(), rdist.sample(), sdist.sample(), n_strokes[b], draw_axis=ax[b, j])
                        ax[b, j].invert_yaxis()
                plt.savefig(f'junks/{e}_{i}.png')
                plt.close()

            out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), ctrlpts, ratws, starts)

            loss = []
            for mu_, std_, mix_, b_, c, r, s, b, l in zip(out_param_mu, out_param_std, out_param_mix, out_stopbits,
                                                         ctrlpts,     ratws,     starts,     stopbits, n_strokes):
                if l >= 2:
                    c, r, s, b = c[1:l.item(), ...], r[1:l.item(), ...], s[1:l.item(), ...], b[1:l.item(), ...]
                    mu_, std_, mix_, b_ = mu_[:l.item()-1, ...], std_[:l.item()-1, ...], mix_[:l.item()-1, ...], b_[:l.item()-1, ...]
                    gmml = gmm_loss(mu_, std_, mix_, args.n_mix, c, r, s)
                    loss.append( gmml + (-b*torch.log(b_)).mean())

            loss = sum(loss) / len(loss)

            if i % args.interval == 0:
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {loss:.4f}')
                writer.add_scalar('train-loss', loss.item(), global_step=count)
                count += 1

            optim.zero_grad()
            loss.backward()
            optim.step()

        # evaluation phase
        avg_loss = 0.
        model.eval()
        for i, B in enumerate(qdltest):
            with torch.no_grad():
                ctrlpts, ratws, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)
            
            out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), ctrlpts, ratws, starts)

            loss = []
            for mu_, std_, mix_, b_, c, r, s, b, l in zip(out_param_mu, out_param_std, out_param_mix, out_stopbits,
                                                         ctrlpts,     ratws,     starts,     stopbits, n_strokes):
                if l >= 2:
                    c, r, s, b = c[1:l.item(), ...], r[1:l.item(), ...], s[1:l.item(), ...], b[1:l.item(), ...]
                    mu_, std_, mix_, b_ = mu_[:l.item()-1, ...], std_[:l.item()-1, ...], mix_[:l.item()-1, ...], b_[:l.item()-1, ...]
                    gmml = gmm_loss(mu_, std_, mix_, args.n_mix, c, r, s)
                    loss.append( gmml + (-b*torch.log(b_)).mean())

            loss = sum(loss) / len(loss)

            avg_loss = ((avg_loss * i) + loss.item()) / (i + 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f'[Testing: -/{e}/{args.epochs}] -> Loss: {avg_loss:.4f}')
            torch.save(model.state_dict(), os.path.join(args.base, args.modelname))
            writer.add_scalar('test-loss', avg_loss, global_step=e)
            
        # savefile = os.path.join(args.base, 'logs', args.tag, str(e) + '.png')
        # inference(qdtest.get_dataloader(args.batch_size), model, embedder, emblayers=args.emblayers, embhidden=args.embhidden,
        #     layers=args.layers, hidden=args.hidden, variational=False, bezier_degree=args.bezier_degree,
        #     nsamples=args.nsamples, rsamples=args.rsamples, savefile=savefile)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')
    parser.add_argument('--n_classes', '-c', type=int, required=False, default=3, help='no. of classes')
    parser.add_argument('--raw', action='store_true', help='Use raw QuickDraw data')
    parser.add_argument('--npz', action='store_true', help='Use .npz QuickDraw data')
    parser.add_argument('--max_sketches_each_cat', '-n', type=int, required=False, default=25000, help='Max no. of sketches each category')

    parser.add_argument('--embvariational', action='store_true', help='Impose prior on latent space (in embedder)')
    parser.add_argument('--embhidden', type=int, required=False, default=16, help='no. of hidden neurons (in embedder)')
    parser.add_argument('--emblayers', type=int, required=False, default=1, help='no of layers (in embedder)')
    parser.add_argument('--embmodel', type=str, required=True, help='path to the pre-trained embedder')
    parser.add_argument('-T', '--stochastic_t', action='store_true', help='Use stochastic t-values')
    parser.add_argument('--hidden', type=int, required=False, default=256, help='no. of hidden neurons')
    parser.add_argument('-x', '--n_mix', type=int, required=False, default=3, help='no. of GMM mixtures')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in encoder RNN')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=5, help='degree of the bezier')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--dropout', type=float, required=False, default=0.8, help='Dropout rate')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    # parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('--rendersketch', action='store_true', help='Render the sketches (debugging purpose)')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-i', '--interval', type=int, required=False, default=50, help='logging interval')
    parser.add_argument('--nsamples', type=int, required=False, default=2, help='no. of data samples for inference')
    parser.add_argument('--rsamples', type=int, required=False, default=1, help='no. of distribution samples for inference')
    args = parser.parse_args()

    main( args )