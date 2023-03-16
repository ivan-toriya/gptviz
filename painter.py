import pandas as pd
import openai

class Strategy():
    def __init__(self, df, dataset_name):
        self.df = df
        self.dataset_name = dataset_name

    def get_response(self):
        pass

    def execute(self, prompt_input):
        pass
    
    def _sample_data(self):
        sample_data = self.df.sample(5).to_csv(index=False)
        sample_data = sample_data.replace("\n", "\n# ")[:-3]
        return sample_data


class CushmanStrategy(Strategy):

    model = "code-cushman-001"
    max_tokens = 512
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    best_of=1,

    def __init__(self, df, dataset_name):
        super().__init__(df, dataset_name)

    def _prompt_header(self):
        header = (
            "# Python 3\n"
            "\n"
            "import pandas as pd\n"
            "import plotly.express as px\n"
            "\n"
            f"df = pd.read_csv(\"{self.dataset_name}.csv\")\n"
            "\n"
            "# csv header and 5 rows of data as an example\n"
            f"# {super()._sample_data()}\n"
            "\n"
            )
        return header

    def _prompt(self):
        prompt = (
            f"{self._prompt_header()}"
            "\"\"\"\n"
            f"{self.prompt_input}\n"
            "\"\"\"\n"
            "def fig():\n"
        )
        return prompt

    def get_response(self):
        output = openai.Completion.create(
            model=self.model,
            prompt=self._prompt(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            best_of=self.best_of,
            echo=False,
            stop=["return fig"]
        )

        return output.choices[0].text

    def _code_to_exec(self):
        code = (
            "def fig():\n"
            f"    {self.get_response()}"
            f"return fig"
            f"\n\nfig = fig()"
        )

    def execute(self, prompt_input):
        print("CushmanStrategy")



class DavinciStrategy(Strategy):
    def execute(self, prompt_input):
        print("DavinciStrategy")


class Visualizer():
    def __init__(self, strategy):
        self.strategy = strategy
    
    def show(self, prompt_input):
        return self.strategy.execute(prompt_input)


if __name__ == "__main__":
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    dataset_name = "custom_data"
    painter = Visualizer(CushmanStrategy(df, dataset_name))
    painter.show("")