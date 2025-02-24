def add_args(parser):
    parser.add_argument("--model_type", help="indo-bart/indo-t5")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--max_seq_len_amr", type=int, default=512)
    parser.add_argument("--max_seq_len_sent", type=int, default=384)
    parser.add_argument("--result_folder", default='result')
    parser.add_argument("--data_folder", default='../data/preprocessed_data/linearized_penman')
    parser.add_argument("--saved_model_folder_path")
    parser.add_argument('--resume_from_checkpoint', default=False)

    return parser

    