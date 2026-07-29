import os
import re
import argparse

import pandas as pd
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate



CONFIG = {
    "dataset": {
        "input_path": "sample_data/cars_source.csv",
        "output_path": "sample_data/cars_features.csv",
        "id_column": "id",
        "car_name_column": "car_name",
        "car_model_column": "car_model",
        "car_year_column": "car_year",
        "dedupe": True,
    },
    "llm": {
        "provider": "google",  
        "model_priority": ["gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-2.5-flash-lite"],
        "temperature": 0.0,
        "max_tokens": 512,
    },
    "structured_query": {
        "retries": 1,
    },
    "features": [
        {
            "key": "height_cm",
            "type": "numeric",
            "question": "What is the approximate overall height (ground to roof, in centimeters) of a {car_year} {car_name} {car_model}?",
            "choices": [],
        },
        {
            "key": "body_type",
            "type": "choice",
            "question": "What body type best describes the {car_year} {car_name} {car_model}?",
            "choices": ["sedan", "suv", "hatchback", "coupe", "pickup_truck", "van", "wagon", "convertible", "unknown"],
        },
        {
            "key": "length_cm",
            "type": "numeric",
            "question": "What is the approximate overall length (bumper to bumper, in centimeters) of a {car_year} {car_name} {car_model}?",
            "choices": [],
        },
    ],
}



_cached_llm = None


def _build_client(provider, model_name, temperature, max_tokens):
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name, temperature=temperature, max_tokens=max_tokens)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)
    raise ValueError(f"Unsupported provider: {provider}")


def get_llm(config, verbose=True):
    global _cached_llm
    if _cached_llm is not None:
        return _cached_llm

    llm_cfg = config["llm"]
    last_error = None

    for model_name in llm_cfg["model_priority"]:
        try:
            client = _build_client(
                llm_cfg["provider"], model_name,
                llm_cfg.get("temperature", 0.0), llm_cfg.get("max_tokens", 512),
            )
            client.invoke("ping")
            if verbose:
                print(f"[get_llm] Using model: {model_name}")
            _cached_llm = client
            return client
        except Exception as e:
            last_error = e
            if verbose:
                print(f"[get_llm] '{model_name}' unavailable ({e}), trying next...")

    raise RuntimeError(f"No working model found. Last error: {last_error}")


structured_prompt = PromptTemplate(
    input_variables=["question", "choices_block"],
    template=(
        "You are a precise automotive data assistant.\n"
        "Question: {question}\n"
        "{choices_block}"
        "Respond with ONLY the answer value, no explanation, no units unless asked."
    ),
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


class ChoiceParser(BaseOutputParser):
    choices: list

    def parse(self, text):
        cleaned = text.strip().strip(".").lower()
        for c in self.choices:
            if c.lower() == cleaned or c.lower() in cleaned:
                return c
        raise OutputParserException(f"'{text}' does not match any choice in {self.choices}")


class NumericParser(BaseOutputParser):
    def parse(self, text):
        match = _NUMBER_RE.search(text.replace(",", ""))
        if not match:
            raise OutputParserException(f"No number found in '{text}'")
        return float(match.group())


qa_parser = NumericParser()


def make_qa_parser(feature):
    if feature["type"] == "choice":
        return ChoiceParser(choices=feature["choices"])
    return NumericParser()


def _get_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _choices_block(choices):
    if not choices:
        return ""
    return f"Choose exactly one of: {', '.join(choices)}\n"


def ask_structured(question, choices, llm, parser=qa_parser, prompt=structured_prompt, retries=1):
    rendered = prompt.format(question=question, choices_block=_choices_block(choices))
    raw = None
    text = None

    for attempt in range(retries + 1):
        raw = llm.invoke(rendered)
        text = _get_text(raw.content)
        try:
            return parser.parse(text), text
        except OutputParserException as e:
            print(f"[ask_structured] Parse failure on attempt {attempt}: {e}")

    print("[ask_structured] Falling back to self-repair...")
    from langchain.output_parsers import OutputFixingParser
    fixer = OutputFixingParser.from_llm(parser=parser, llm=llm)
    return fixer.parse(text), text



def build_questions_for_car(config, car_name, car_model, car_year=""):
    questions = []
    for feature in config["features"]:
        text = feature["question"].format(car_name=car_name, car_model=car_model, car_year=car_year).strip()
        questions.append((feature, text))
    return questions


def feature_keys(config):
    return [f["key"] for f in config["features"]]




def _read_source(config):
    ds = config["dataset"]
    df = pd.read_csv(ds["input_path"])

    required = [ds["car_name_column"], ds["car_model_column"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if ds["id_column"] not in df.columns:
        df[ds["id_column"]] = range(1, len(df) + 1)

    return df


def _unique_cars(df, config):
    ds = config["dataset"]
    key_cols = [ds["car_name_column"], ds["car_model_column"]]
    year_col = ds.get("car_year_column")
    if year_col and year_col in df.columns:
        key_cols.append(year_col)

    if not ds.get("dedupe", True):
        return df[[ds["id_column"]] + key_cols].copy()

    return df.drop_duplicates(subset=key_cols, keep="first")[[ds["id_column"]] + key_cols].reset_index(drop=True)


def run_pipeline(config, llm=None, verbose=True):
    ds = config["dataset"]
    llm = llm or get_llm(config, verbose=verbose)

    source_df = _read_source(config)
    cars_df = _unique_cars(source_df, config)
    year_col = ds.get("car_year_column")

    keys = feature_keys(config)
    feature_parsers = {f["key"]: make_qa_parser(f) for f in config["features"]}
    retries = config.get("structured_query", {}).get("retries", 1)

    rows = []
    for _, row in cars_df.iterrows():
        car_id = row[ds["id_column"]]
        car_name = row[ds["car_name_column"]]
        car_model = row[ds["car_model_column"]]
        car_year = row[year_col] if year_col and year_col in row else ""

        if verbose:
            print(f"\n[pipeline] {car_id}: {car_name} {car_model} {car_year}".strip())

        record = {"id": car_id, "car_name": car_name, "car_model": car_model}
        if year_col:
            record["car_year"] = car_year

        for feature, question in build_questions_for_car(config, str(car_name), str(car_model), str(car_year or "")):
            parser = feature_parsers[feature["key"]]
            try:
                value, raw_text = ask_structured(
                    question=question, choices=feature.get("choices", []),
                    llm=llm, parser=parser, prompt=structured_prompt, retries=retries,
                )
                if verbose:
                    print(f"    {feature['key']} -> {value}")
            except OutputParserException as e:
                print(f"    [WARN] {feature['key']} failed: {e}")
                value = None
            record[feature["key"]] = value

        rows.append(record)

    columns = ["id", "car_name", "car_model"] + (["car_year"] if year_col else []) + keys
    output_df = pd.DataFrame(rows, columns=columns)

    os.makedirs(os.path.dirname(ds["output_path"]) or ".", exist_ok=True)
    output_df.to_csv(ds["output_path"], index=False)

    if verbose:
        print(f"\n[pipeline] Wrote {len(output_df)} rows -> {ds['output_path']}")

    return output_df






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=None, help="Override dataset.input_path")
    parser.add_argument("--output", default=None, help="Override dataset.output_path")
    args = parser.parse_args()

    if args.input:
        CONFIG["dataset"]["input_path"] = args.input
    if args.output:
        CONFIG["dataset"]["output_path"] = args.output

    llm = get_llm(CONFIG)
    output_df = run_pipeline(CONFIG, llm=llm)

    print("\n=== Sample of output dataset ===")
    print(output_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
