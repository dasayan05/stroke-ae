import matplotlib.pyplot as plt
import torch, os, numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils import tensorboard as tb

from quickdraw.quickdraw import QuickDraw
from bezierae import RNNBezierAE, RNNSketchAE
from beziercurve import draw_bezier

def drawsketch(ctrlpts, ratws, st_starts, n_stroke):
    ctrlpts, ratws, st_starts = ctrlpts[:n_stroke], ratws[:n_stroke], st_starts[:n_stroke]
    ctrlpts = ctrlpts.view(-1, ctrlpts.shape[-1] // 2, 2)
    fig = plt.figure()
    for ctrlpt, ratw, st_start in zip(ctrlpts, ratws, st_starts):
        ctrlpt = ctrlpt.detach().cpu().numpy()
        ratw = ratw.detach().cpu().numpy()
        st_start = st_start.detach().cpu().numpy()
        draw_bezier(ctrlpt, ratw, start_xy=st_start, draw_axis=plt.gca(),
            ctrlPointPlotKwargs=dict(marker='X', color='red', linestyle='--', alpha=0.4))
    plt.gca().invert_yaxis()
    plt.show()

def stroke_embed(batch, initials, embedder, viz = False):
    h_initial, c_initial = initials
    # Redundant, but thats fine
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # accumulate all info into these empty lists
    sketches_ctrlpt, sketches_ratw, sketches_st_starts, sketches_stopbits = [], [], [], []
    n_strokes = []
    
    for sk, _ in batch:
        # for each sketch in the batch
        st_starts = torch.tensor([st[0,:2] for st in sk], device=device)
        sk = [torch.tensor(st[:,:-1], device=device) - st_start for st, st_start in zip(sk, st_starts)]
        ls = [st.shape[0] for st in sk]
        sk = pad_sequence(sk, batch_first=True)
        sk = pack_padded_sequence(sk, ls, batch_first=True, enforce_sorted=False)
        emb_ctrlpt, emb_ratw = embedder(sk, h_initial, c_initial)
        sketches_ctrlpt.append(emb_ctrlpt.view(len(ls), -1))
        sketches_ratw.append(emb_ratw)
        sketches_st_starts.append(st_starts)
        # create stopbits
        stopbit = torch.zeros(emb_ctrlpt.shape[0], 1, device=device); stopbit[-1, 0] = 1.
        sketches_stopbits.append(stopbit)
        n_strokes.append(emb_ctrlpt.shape[0])
        
        if viz: # optionally visualize "per-stroke bezier representation" of a sketch
            drawsketch(sketches_ctrlpt[-1], sketches_ratw[-1], sketches_st_starts[-1], n_strokes[-1])
    
    n_strokes = torch.tensor(n_strokes, device=device)
    sketches_ctrlpt = pad_sequence(sketches_ctrlpt, batch_first=True)
    sketches_ratw = pad_sequence(sketches_ratw, batch_first=True)
    sketches_st_starts = pad_sequence(sketches_st_starts, batch_first=True)
    sketches_stopbits = pad_sequence(sketches_stopbits, batch_first=True)

    # For every sketch in a batch:
    #   For every stroke in the sketch:
    #     1. (Control Point, Rational Weights) pair
    #     2. Start location of the stroke with respect to a global reference (of the sketch)
    return sketches_ctrlpt, sketches_ratw, sketches_st_starts, sketches_stopbits, n_strokes

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'mosquito', 'firetruck', 'owl', 'pig', 'face', 'purse', 'shoe' ]

    qd = QuickDraw(args.root, categories=chosen_classes[:args.n_classes], max_sketches_each_cat=args.max_sketches_each_cat, verbose=True, normalize_xy=True, start_from_zero=False, mode=QuickDraw.STROKESET, raw=args.raw)
    
    qdtrain, qdtest = qd.split(0.8)
    qdltrain, qdltest = qdtrain.get_dataloader(args.batch_size), qdtest.get_dataloader(args.batch_size)

    # chosen device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Embedder model (pretrained and freezed)
    embedder = RNNBezierAE(2, args.embhidden, args.emblayers, args.bezier_degree, bidirectional=True, variational=args.embvariational)
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
    n_ratw = args.bezier_degree + 1
    n_ctrlpt = n_ratw * 2
    model = RNNSketchAE((n_ctrlpt, n_ratw, 2), args.hidden, dropout=args.dropout)
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

            out_ctrlpts, out_ratws, out_starts, out_stopbits = model((h_initial, c_initial), ctrlpts, ratws, starts)

            loss = []
            for c_, r_, s_, b_, c, r, s, b, l in zip(out_ctrlpts, out_ratws, out_starts, out_stopbits,
                                                         ctrlpts,     ratws,     starts,     stopbits, n_strokes):
                if l >= 2:

                    c, r, s, b = c[1:l.item(), ...], r[1:l.item(), ...], s[1:l.item(), ...], b[1:l.item(), ...]
                    c_, r_, s_, b_ = c_[:l.item()-1, ...], r_[:l.item()-1, ...], s_[:l.item()-1, ...], b_[:l.item()-1, ...]

                    loss.append( ((c - c_) ** 2).mean() + ((r - r_) ** 2).mean() + ((s - s_) ** 2).mean() + ((b - b_) ** 2).mean() )

            loss = sum(loss) / len(loss)

            if i % args.interval == 0:
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {loss:.4f}')
                writer.add_scalar('train-loss', loss.item(), global_step=count)
                count += 1

            optim.zero_grad()
            loss.backward()
            optim.step()

        # evaluation phase
        model.eval()
        avg_loss = 0.
        for i, B in enumerate(qdltest):
            with torch.no_grad():
                ctrlpts, ratws, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)

            out_ctrlpts, out_ratws, out_starts, out_stopbits = model((h_initial, c_initial), ctrlpts, ratws, starts)

            loss = []
            for c_, r_, s_, b_, c, r, s, b, l in zip(out_ctrlpts, out_ratws, out_starts, out_stopbits,
                                                         ctrlpts,     ratws,     starts,     stopbits, n_strokes):
                if l >= 2:

                    c, r, s, b = c[1:l.item(), ...], r[1:l.item(), ...], s[1:l.item(), ...], b[1:l.item(), ...]
                    c_, r_, s_, b_ = c_[:l.item()-1, ...], r_[:l.item()-1, ...], s_[:l.item()-1, ...], b_[:l.item()-1, ...]

                    loss.append( ((c - c_) ** 2).mean() + ((r - r_) ** 2).mean() + ((s - s_) ** 2).mean() + ((b - b_) ** 2).mean() )

            loss = sum(loss) / len(loss)

            avg_loss = ((avg_loss * i) + loss.item()) / (i + 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            print(f'[Testing: -/{e}/{args.epochs}] -> Loss: {avg_loss:.4f}')
            torch.save(model.state_dict(), os.path.join(args.base, args.modelname))
            writer.add_scalar('test-loss', avg_loss, global_step=e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')
    parser.add_argument('--n_classes', '-c', type=int, required=False, default=3, help='no. of classes')
    parser.add_argument('--raw', action='store_true', help='Use raw QuickDraw data')
    parser.add_argument('--max_sketches_each_cat', '-n', type=int, required=False, default=25000, help='Max no. of sketches each category')

    parser.add_argument('--embvariational', action='store_true', help='Impose prior on latent space (in embedder)')
    parser.add_argument('--embhidden', type=int, required=False, default=16, help='no. of hidden neurons (in embedder)')
    parser.add_argument('--emblayers', type=int, required=False, default=1, help='no of layers (in embedder)')
    parser.add_argument('--embmodel', type=str, required=True, help='path to the pre-trained embedder')
    parser.add_argument('--hidden', type=int, required=False, default=256, help='no. of hidden neurons')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in encoder RNN')
    parser.add_argument('-z', '--bezier_degree', type=int, required=False, default=5, help='degree of the bezier')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--dropout', type=float, required=False, default=0.8, help='Dropout rate')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    # parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    # parser.add_argument('--regp', type=float, required=False, default=1e-2, help='Regularizer weight on control points')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('-i', '--interval', type=int, required=False, default=50, help='logging interval')
    # parser.add_argument('--nsample', type=int, required=False, default=6, help='no. of data samples for inference')
    # parser.add_argument('--rsample', type=int, required=False, default=6, help='no. of distribution samples for inference')
    args = parser.parse_args()

    main( args )