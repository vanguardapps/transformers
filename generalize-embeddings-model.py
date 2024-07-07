from datasets import load_dataset
from transformers.generation import KNNStoreSQLite


def main():
    checkpoint = "facebook/nllb-200-distilled-600M"
    src_lang = "eng_Latn"
    tgt_lang = "deu_Latn"
    # batch_size = 10

    # dataset = load_dataset("csv", data_files="data/de-en-emea-medical-clean.csv")[
    #     'train'
    # ]

    knn_store = KNNStoreSQLite(
        checkpoint=checkpoint,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
        database="db/test_db.db",
    )

    # knn_store.truncate_artifacts(remove_all=True)

    # knn_store.ingest(
    #     dataset,
    #     src_col="english",
    #     tgt_col="german",
    #     batch_size=batch_size,
    #     progress_bar=True,
    # )

    knn_store.build_source_index(progress_bar=True)

    # # AFTER doing all of the above work, the below should "just work"
    # model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    # tokenizer = AutoTokenizer.from_pretrained(
    #     checkpoint, src_lang=src_lang, tgt_lang=tgt_lang
    # )

    # output_ids = model.generate(
    #     **tokenizer(
    #         [
    #             "I am doing fine",
    #             "Hello, how are you",
    #             "What is the best antibiotic?",
    #             "Where are my pants?",
    #             "I like boobs.",
    #         ],
    #         padding='max_length',
    #         truncation=True,
    #         max_length=150,
    #         return_tensors="pt",
    #     ),
    #     knn_store=knn_store,
    #     knn_interpolation_coefficient=0.5,
    #     forced_bos_token_id=tokenizer.convert_tokens_to_ids([tgt_lang])[0],
    # )

    # print(output_ids)
    # print(tokenizer.batch_decode(output_ids))


if __name__ == "__main__":
    main()
