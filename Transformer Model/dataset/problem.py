import bert_data
def prepare(problem_set, data_dir, max_length, batch_size, device, opt):
    setattr(opt, 'share_target_embedding', False)
    setattr(opt, 'has_inputs', True)

    train_iter, val_iter, opt = \
        bert_data.prepare(max_length, batch_size, device, opt, data_dir)

    return train_iter, val_iter, opt.src_vocab_size, opt.trg_vocab_size, opt