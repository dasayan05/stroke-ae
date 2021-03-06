import matplotlib.pyplot as plt
import torch, os, numpy as np
from torch.distributions import Normal
from torch.utils import tensorboard as tb

from quickdraw.quickdraw import QuickDraw
from bezierae import RNNBezierAE, RNNSketchAE, gmm_loss
from infer_beziersketch import inference, drawsketch, stroke_embed
from npz import NPZWriter

def select_degree(ctrlpts, deg_loss):
    batch = []
    for cpts, degloss in zip(ctrlpts, deg_loss):
        sketch = []
        for i_stroke, dloss in enumerate(degloss):
            opt_degree = (dloss < 5e-5).nonzero()[0]
            if opt_degree.size != 0:
                opt_degree = opt_degree[0]
            else:
                opt_degree = len(dloss) - 1

            t = cpts[opt_degree][i_stroke,:]
            sketch.append( t )
        batch.append(sketch)
    return batch

def main( args ):
    chosen_classes = [ 'cat', 'chair', 'mosquito', 'firetruck', 'owl', 'pig', 'face', 'purse', 'shoe' ]
    if args.iam:
        chosen_classes = ['iam']

    qd = QuickDraw(args.root, categories=[chosen_classes[args.n_class],], max_sketches_each_cat=args.max_sketches_each_cat,
        verbose=True, normalize_xy=True, start_from_zero=False, mode=QuickDraw.STROKESET, raw=args.raw, npz=args.npz)
    
    qdtrain, qdtest = qd.split(0.8)
    qdltrain = qdtrain.get_dataloader(args.batch_size)

    # chosen device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Embedder model (pretrained and freezed)
    embedder = RNNBezierAE(2, args.embhidden, args.emblayers, args.emblatent, args.bezier_degree_low, args.bezier_degree_high,
        bidirectional=True, rational=args.rational)
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
    n_ctrlpt = (args.bezier_degree + 1 - 1) * 2
    model = RNNSketchAE((n_ctrlpt, n_ratw, 2), args.hidden, dropout=args.dropout, n_mixture=args.n_mix,
        rational=args.rational, variational=args.variational, concatz=args.concatz)
    
    h_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
    c_initial = torch.zeros(args.layers * 2, args.batch_size, args.hidden, dtype=torch.float32)
    if torch.cuda.is_available():
        model, h_initial, c_initial = model.cuda(), h_initial.cuda(), c_initial.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)

    writer = tb.SummaryWriter(os.path.join(args.base, 'logs', args.tag))
    npzwriter = NPZWriter(args.npzfile)
    linear = lambda e, e0, T: max(min((e - e0) / float(T), 1.), 0.)

    count, best_loss = 0, np.inf
    for e in range(args.epochs):
        model.train()
        for i, B in enumerate(qdltrain):
            with torch.no_grad():
                if args.rational:
                    ctrlpts, ratws, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder, args.bezier_degree, args.bezier_degree_low)
                else:
                    ctrlpts, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder, args.bezier_degree, args.bezier_degree_low,
                        inf_loss=args.producenpz or args.rendersketch)
                    ratws = torch.ones(args.batch_size, starts.shape[1], n_ratw, device=device) # FAKE IT
                    if e == 0 and args.producenpz:
                        ctrlpts = select_degree(*ctrlpts)
                        ## DO THIS
                        npzwriter.add(ctrlpts, starts, n_strokes)
                        if i % 10 == 0:
                            npzwriter.flush()
                        continue
            
            if args.rendersketch:
                ctrlpts = select_degree(*ctrlpts)
                for b in range(len(ctrlpts)):
                    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
                    if b > 20:
                        break
                    drawsketch(ctrlpts[b], ratws[b], starts[b], n_strokes[b], draw_axis=ax, invert_y=not args.raw)
                    ax.set_xticks([]); ax.set_yticks([])
                    plt.savefig(f'junks/{e}_{i}_{b}.png', bbox_inches='tight', inches=0)
                    plt.close()
                continue


            _cpad = torch.zeros(ctrlpts.shape[0], 1, ctrlpts.shape[2], device=device)
            _rpad = torch.zeros(ratws.shape[0], 1, ratws.shape[2], device=device)
            _spad = torch.zeros(starts.shape[0], 1, starts.shape[2], device=device)
            _bpad = torch.zeros(stopbits.shape[0], 1, stopbits.shape[2], device=device)
            ctrlpts, ratws, starts, stopbits = torch.cat([_cpad, ctrlpts], dim=1), \
                                    torch.cat([_rpad, ratws], dim=1), \
                                    torch.cat([_spad, starts], dim=1), \
                                    torch.cat([_bpad, stopbits], dim=1)

            if args.rational:
                out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), ctrlpts, ratws, starts)
            else:
                if args.variational:
                    out_param_mu, out_param_std, out_param_mix, out_stopbits, KLD = model((h_initial, c_initial), ctrlpts, None, starts)
                    ann = linear(e, 0, 10)
                else:
                    out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), ctrlpts, None, starts)
                    KLD = 0.
                    ann = 0.
            
            loss = []
            for mu_, std_, mix_, b_, c, r, s, b, l in zip(out_param_mu, out_param_std, out_param_mix, out_stopbits,
                                                         ctrlpts,     ratws,     starts,     stopbits, n_strokes):
                if l >= 1:
                    c, r, s, b = c[1:l.item()+1, ...], r[1:l.item()+1, ...], s[1:l.item()+1, ...], b[1:l.item()+1, ...]
                    mu_, std_, mix_, b_ = mu_[:l.item(), ...], std_[:l.item(), ...], mix_[:l.item(), ...], b_[:l.item(), ...]
                    # preparing for mdn loss calc
                    mu_ = mu_.view(1, l.item(), args.n_mix, -1)
                    std_ = std_.view(1, l.item(), args.n_mix, -1)
                    if args.rational:
                        param_ = torch.cat([c, r, s], -1).view(1, l.item(), -1)
                    else:
                        param_ = torch.cat([c, s], -1).view(1, l.item(), -1)
                    mix_ = mix_.log().view(1, l.item(), args.n_mix)
                    gmml = gmm_loss(param_, mu_, std_, mix_, reduce=True)
                    stopbitloss = (b - b_).pow(2).mean()
                    loss.append( gmml + stopbitloss )

            recon = sum(loss) / len(loss)
            loss = recon + KLD * args.wkl * ann

            if i % args.interval == 0:
                print(f'[Training: {i}/{e}/{args.epochs}] -> Loss: {recon:.4f} + {ann:.4f} x {KLD:.4f} = {loss:.4f}')
                writer.add_scalar('train-loss', loss.item(), global_step=count)
                count += 1

            optim.zero_grad()
            loss.backward()
            optim.step()

        # flush the npz
        if e == 0 and args.producenpz:
            npzwriter.flush()
            exit()

        # # evaluation phase
        # avg_loss = 0.
        model.eval()
        # for i, B in enumerate(qdltest):
        #     with torch.no_grad():
        #         if args.rational:
        #             ctrlpts, ratws, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)
        #         else:
        #             ctrlpts, starts, stopbits, n_strokes = stroke_embed(B, (h_initial_emb, c_initial_emb), embedder)
        #             ratws = torch.ones(args.batch_size, ctrlpts.shape[1], n_ratw, device=device) # FAKE IT
            
        #     if args.rational:
        #         out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), ctrlpts, ratws, starts)
        #     else:
        #         if args.variational:
        #             out_param_mu, out_param_std, out_param_mix, out_stopbits, KLD = model((h_initial, c_initial), ctrlpts, None, starts)
        #             ann = linear(e, 0, 10)
        #         else:
        #             out_param_mu, out_param_std, out_param_mix, out_stopbits = model((h_initial, c_initial), ctrlpts, None, starts)
        #             KLD = 0.
        #             ann = 0.

        #     _cpad = torch.zeros(ctrlpts.shape[0], 1, ctrlpts.shape[2], device=device)
        #     _rpad = torch.zeros(ratws.shape[0], 1, ratws.shape[2], device=device)
        #     _spad = torch.zeros(starts.shape[0], 1, starts.shape[2], device=device)
        #     _bpad = torch.zeros(stopbits.shape[0], 1, stopbits.shape[2], device=device)
        #     ctrlpts, ratws, starts, stopbits = torch.cat([_cpad, ctrlpts], dim=1), \
        #                             torch.cat([_rpad, ratws], dim=1), \
        #                             torch.cat([_spad, starts], dim=1), \
        #                             torch.cat([_bpad, stopbits], dim=1)

        #     loss = []
        #     for mu_, std_, mix_, b_, c, r, s, b, l in zip(out_param_mu, out_param_std, out_param_mix, out_stopbits,
        #                                                  ctrlpts,     ratws,     starts,     stopbits, n_strokes):
        #         if l >= 1:
        #             c, r, s, b = c[1:l.item()+1, ...], r[1:l.item()+1, ...], s[1:l.item()+1, ...], b[1:l.item()+1, ...]
        #             mu_, std_, mix_, b_ = mu_[:l.item(), ...], std_[:l.item(), ...], mix_[:l.item(), ...], b_[:l.item(), ...]
        #             # preparing for mdn loss calc
        #             mu_ = mu_.view(1, l.item(), args.n_mix, -1)
        #             std_ = std_.view(1, l.item(), args.n_mix, -1)
        #             if args.rational:
        #                 param_ = torch.cat([c, r, s], -1).view(1, l.item(), -1)
        #             else:
        #                 param_ = torch.cat([c, s], -1).view(1, l.item(), -1)
        #             mix_ = mix_.log().view(1, l.item(), args.n_mix)
        #             gmml = gmm_loss(param_, mu_, std_, mix_, reduce=True)
        #             stopbitloss = (-b*torch.log(b_)).mean()
        #             loss.append( gmml + stopbitloss )

        #     loss = sum(loss) / len(loss) + KLD * args.wkl * ann

        #     avg_loss = ((avg_loss * i) + loss.item()) / (i + 1)

        # print(f'[Testing: -/{e}/{args.epochs}] -> Loss: {avg_loss:.4f}')
        # writer.add_scalar('test-loss', avg_loss, global_step=e)
        torch.save(model.state_dict(), os.path.join(args.base, args.modelname))
            
        savefile = os.path.join(args.base, 'logs', args.tag, str(e) + '.png')
        inference(qdtest.get_dataloader(args.batch_size), model, embedder, emblayers=args.emblayers, embhidden=args.embhidden,
            layers=args.layers, hidden=args.hidden, variational=False, bezier_degree=args.bezier_degree, bezier_degree_low=args.bezier_degree_low,
            n_mix=args.n_mix, nsamples=args.nsamples, rsamples=args.rsamples, savefile=savefile, device=device, invert_y=not args.raw)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, help='quickdraw binary file')
    parser.add_argument('--base', type=str, required=False, default='.', help='base folder of operation (needed for condor)')
    parser.add_argument('--n_class', '-c', type=int, required=False, default=0, help='no. of classes')
    parser.add_argument('--iam', action='store_true', help='Use IAM dataset')
    parser.add_argument('--raw', action='store_true', help='Use raw QuickDraw data')
    parser.add_argument('--npz', action='store_true', help='Use .npz QuickDraw data')
    parser.add_argument('--max_sketches_each_cat', '-n', type=int, required=False, default=25000, help='Max no. of sketches each category')

    parser.add_argument('--embvariational', action='store_true', help='Impose prior on latent space (in embedder)')
    parser.add_argument('--embhidden', type=int, required=False, default=16, help='no. of hidden neurons (in embedder)')
    parser.add_argument('--emblayers', type=int, required=False, default=1, help='no of layers (in embedder)')
    parser.add_argument('--emblatent', type=int, required=False, default=256, help='dim of latent vector (in embedder)')
    parser.add_argument('--embmodel', type=str, required=True, help='path to the pre-trained embedder')
    parser.add_argument('-T', '--stochastic_t', action='store_true', help='Use stochastic t-values')
    parser.add_argument('-R', '--rational', action='store_true', help='Rational bezier curve ?')
    parser.add_argument('--concatz', action='store_true', help='concat z with all inputs in decoder')
    parser.add_argument('--hidden', type=int, required=False, default=256, help='no. of hidden neurons')
    parser.add_argument('-x', '--n_mix', type=int, required=False, default=3, help='no. of GMM mixtures')
    parser.add_argument('--layers', type=int, required=False, default=2, help='no of layers in encoder RNN')
    
    parser.add_argument('--bezier_degree', type=int, required=False, default=9, help='degree of the bezier')
    parser.add_argument('-y', '--bezier_degree_low', type=int, required=False, default=9, help='lowest degree of the bezier')
    parser.add_argument('-z', '--bezier_degree_high', type=int, required=False, default=9, help='highest degree of the bezier')

    parser.add_argument('-V', '--variational', action='store_true', help='Impose prior on latent space')
    parser.add_argument('--wkl', type=float, required=False, default=1.0, help='weight of the KL term')
    
    parser.add_argument('-b','--batch_size', type=int, required=False, default=128, help='batch size')
    parser.add_argument('--dropout', type=float, required=False, default=0.8, help='Dropout rate')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='learning rate')
    parser.add_argument('-e', '--epochs', type=int, required=False, default=40, help='no of epochs')
    # parser.add_argument('--anneal_KLD', action='store_true', help='Increase annealing factor of KLD gradually')
    
    parser.add_argument('--tag', type=str, required=False, default='main', help='run identifier')
    parser.add_argument('--rendersketch', action='store_true', help='Render the sketches (debugging purpose)')
    parser.add_argument('-m', '--modelname', type=str, required=False, default='model', help='name of saved model')
    parser.add_argument('--npzfile', type=str, required=False, default='ctrlpt.npz', help='SketchRNN style .npz for control points')
    parser.add_argument('--producenpz', action='store_true', help='Produce npz')
    parser.add_argument('-i', '--interval', type=int, required=False, default=50, help='logging interval')
    parser.add_argument('--nsamples', type=int, required=False, default=6, help='no. of data samples for inference')
    parser.add_argument('--rsamples', type=int, required=False, default=5, help='no. of distribution samples for inference')
    args = parser.parse_args()

    main( args )