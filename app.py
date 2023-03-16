import os
from collections import deque

import openai
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from tenacity import (  # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from get_api_status import get_api_status

load_dotenv()


@st.cache_data
@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(2),
    reraise=True,
)
def get_completion_response(**kwargs):
    return openai.Completion.create(**kwargs)


st.set_page_config(
    page_title="gptviz",
    page_icon="📊",
)

# Header

st.title("GPTViz")
st.text("Data visualization using natural language prompts")
st.text("Supported libraries: Plotly")
st.markdown("---")


# Authentication
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("OPENAI_ORGANIZATION")

st.markdown(f"[OpenAI API](https://status.openai.com/): {get_api_status()}")
choice = st.radio(
    "Model to use",
    (["code-cushman-001", "code-davinci-002 (Recommended)"]),
    index=1,
    help="code-cushman-001 faster, but less accurate; code-davinci-002 slower, but more accurate.",
)  # code-cushman-001 not working with suffixes
MODEL = choice.split(" ")[0]

if st.button("Check connection"):
    st.cache_data.clear()
    prompt = f'>>> print("Hello! I\'m {MODEL} model.")'
    response = get_completion_response(
        model=MODEL,
        prompt=prompt,
        temperature=0,
        top_p=1,
        max_tokens=40,
        best_of=1,
        stop=[">>>", "\n\n", "\n#", "```"],
    )
    if response:
        st.markdown(f"Prompt: `{prompt}`")
        st.markdown("Response: ")
        st.code(response["choices"][0]["text"])
        st.markdown(f"Total tokens used: {response.usage.total_tokens}")


# Data loading


@st.cache_resource
def load_sample_dataset(url, dataset):
    df = pd.read_csv(f"{url}{dataset}.csv")
    return df


@st.cache_data
def load_csv_data(file):
    df = pd.read_csv(file)
    return df


st.subheader("Data input")
tab1, tab2 = st.tabs(["Sample Datasets", "Upload CSV"])

with tab1:
    with st.expander("Dataset settings", expanded=True):
        datasets = {
            "anagrams": "A dataset containing results of an experiment on the time taken to solve anagrams under different conditions.",
            "anscombe": "A dataset containing four sets of data that have the same statistical properties, but different patterns when plotted.",
            "attention": "A dataset containing results of an experiment on how attention affects response time.",
            "brain_networks": "A dataset containing information about the connections between different regions of the human brain.",
            "car_crashes": "A dataset containing information about car crashes, including their severity and factors that may have contributed to them.",
            "diamonds": "A dataset containing information about diamonds, including their carat weight, cut, color, and price.",
            "dots": "A dataset containing results of an experiment on how response time is affected by the number of dots on a screen.",
            "dowjones": "A dataset containing information about daily closing prices of the Dow Jones Industrial Average from 2003 to 2010.",
            "exercise": "A dataset containing information about the duration and type of exercise performed by individuals in a study.",
            "flights": "A dataset containing information about flights departing from New York City in 2013.",
            "fmri": "A dataset containing functional magnetic resonance imaging (fMRI) data from a study on visual object recognition.",
            "geyser": "A dataset containing information about the duration and waiting time between eruptions of the Old Faithful geyser in Yellowstone National Park.",
            "glue": "A dataset containing results of an experiment on the strength of glue under different conditions.",
            "healthexp": "A dataset containing information about health expenditures as a percentage of gross domestic product (GDP) for various countries.",
            "iris": "A dataset containing measurements of iris flowers, often used for classification tasks.",
            "mpg": "A dataset containing information about the fuel efficiency of various car models.",
            "penguins": "A dataset containing measurements of penguin body dimensions and environmental variables.",
            "planets": "A dataset containing information about exoplanets discovered by the Kepler space telescope.",
            "seaice": "A dataset containing information about sea ice extent in the Arctic and Antarctic regions.",
            "taxis": "A dataset containing information about taxi trips in New York City in 2014.",
            "tips": "A dataset containing information about tips given by customers at a restaurant.",
            "titanic": "A dataset containing information about passengers on the Titanic, including their demographics and survival status.",
        }

        dataset = st.selectbox(
            "What dataset do you want to use?",
            (
                "anagrams",
                "anscombe",
                "attention",
                "brain_networks",
                "car_crashes",
                "diamonds",
                "dots",
                "dowjones",
                "exercise",
                "flights",
                "fmri",
                "geyser",
                "glue",
                "healthexp",
                "iris",
                "mpg",
                "penguins",
                "planets",
                "seaice",
                "taxis",
                "tips",
                "titanic",
            ),
            index=14,
        )
        st.caption(f"{datasets[dataset]}")
        df = load_sample_dataset(
            "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/", dataset
        )
        st.dataframe(df)

with tab2:
    csv_file = st.file_uploader("Upload your data", type="csv")
    if csv_file:
        df = load_csv_data(csv_file)
        st.dataframe(df)


# Dialog


@st.cache_data
def draw_plotly_chart(fig):
    st.plotly_chart(fig)


def show_executed_code():
    with st.expander("Generated code", expanded=True):
        st.code(final_code + "\nfig.show()  # st.plotly_chart(fig)")


sample_data = df.sample(5).to_csv(index=False)
sample_data = sample_data.replace("\n", "\n# ")[:-3]

if "undo_code" not in st.session_state:
    st.session_state["undo_code"] = deque()
if "redo_code" not in st.session_state:
    st.session_state["redo_code"] = deque()
if "undo_prompt" not in st.session_state:
    st.session_state["undo_prompt"] = deque()
if "redo_prompt" not in st.session_state:
    st.session_state["redo_prompt"] = deque()

st.subheader("Conversation")
with st.expander("Conversation example", expanded=True):
    st.markdown(
        """
    *Using the iris dataset*
    ```
    You: Create a scatter plot of sepal length and petal length grouped by species

    [Model generates a scatter plot]

    You: Change to histogram

    [Model creates a histogram]

    You: Create a 3d scatter plot

    [Model generates a 3d scatter plot]

    You: Add title \"My lovely iris collection\"

    [Model adds a title]
    ```

    *Using the diamonds dataset*
    ```
    You: Create a histogram of carat

    [Model generates a histogram]

    You: Change to bar chart of cut by price

    [Model generates a bar chart]

    You: Add title \"My lovely diamonds collection\"

    [Model adds a title]
    ```
    """
    )
with st.form("prompt_form", clear_on_submit=True):
    prompt_input = st.text_input("Enter your prompt here")
    submit_button = st.form_submit_button("Submit")
    if submit_button:
        st.session_state.undo_prompt.append(prompt_input)

col1, col2, col3 = st.columns(3)
with col1:
    back_button_pressed = st.button("⬅ Back")
    if back_button_pressed:
        if st.session_state.undo_code:
            st.session_state.redo_code.append(st.session_state.undo_code.pop())
            st.session_state.redo_prompt.append(st.session_state.undo_prompt.pop())
        else:
            st.warning("Nothing to undo_code")
with col2:
    forward_button_pressed = st.button("➡ Forward")
    if forward_button_pressed:
        if st.session_state.redo_code:
            st.session_state.undo_code.append(st.session_state.redo_code.pop())
            st.session_state.undo_prompt.append(st.session_state.redo_prompt.pop())
        else:
            st.warning("Nothing to redo_code")
with col3:
    clear_button_pressed = st.button("Clear conversation")
    if clear_button_pressed:
        st.session_state.undo_code.clear()
        st.session_state.redo_code.clear()
        st.session_state.undo_prompt.clear()
        st.session_state.redo_prompt.clear()
        prompt_input = None

if st.session_state.undo_prompt:
    for prompt in st.session_state.undo_prompt:
        st.text(f"You: {prompt}")
    st.text("Response:")

# Code generation and visualization

if prompt_input and not st.session_state.undo_code and MODEL == "code-davinci-002":

    prompt = f"""
# Python 3

import pandas as pd
import plotly.express as px

df = pd.read_csv("{dataset}.csv")

# csv header and 5 rows of sample data
# {sample_data}

\"\"\"
{prompt_input}
\"\"\"
def fig():

    """
    # st.text("New prompt:\n" + prompt)
    output = get_completion_response(
        model=MODEL,
        prompt=prompt,
        max_tokens=512,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        echo=False,
        # stop = ["return fig"]
        suffix="\n\nfig = fig()\nfig.show()",
    )

    response = output.choices[0].text

    final_code = "def fig():\n"
    final_code += f"    {response}"
    final_code += f"\n\nfig = fig()"
    if submit_button:
        st.session_state.undo_code.append(final_code)
        # st.write(list(st.session_state.undo_code))
        # st.write(list(st.session_state.redo_code))
        exec(st.session_state.undo_code[-1])

        draw_plotly_chart(fig)
        show_executed_code()

elif prompt_input and st.session_state.undo_code and MODEL == "code-davinci-002":
    prompt = f"# Old fig()\n{st.session_state.undo_code[-1]}"
    prompt += "\n\n"
    prompt += "# " + prompt_input + "\n"
    prompt += "# New fig()\n"
    prompt += "def fig():\n"
    # st.text("New prompt:\n" + prompt)
    output = get_completion_response(
        model=MODEL,
        prompt=prompt,
        max_tokens=512,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        echo=False,
        suffix="\n\nfig = fig()\nfig.show()",
    )
    response = output.choices[0].text
    final_code = "def fig():\n"
    final_code += f"{response}"
    final_code += f"\n\nfig = fig()"

    if submit_button:
        st.session_state.undo_code.append(final_code)
    # st.write(list(st.session_state.undo_code))
    # st.write(list(st.session_state.redo_code))
    exec(st.session_state.undo_code[-1])

    draw_plotly_chart(fig)
    show_executed_code()

if prompt_input and not st.session_state.undo_code and MODEL == "code-cushman-001":

    prompt = f"""
# Python 3

import pandas as pd
import plotly.express as px

df = pd.read_csv("{dataset}.csv")

# csv header and 5 rows of sample data
# {sample_data}

\"\"\"
{prompt_input}
\"\"\"
def fig():

    """
    # st.text("New prompt:\n" + prompt)
    output = get_completion_response(
        model=MODEL,
        prompt=prompt,
        max_tokens=512,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        echo=False,
        stop=["return fig"],
    )

    response = output.choices[0].text

    final_code = "def fig():\n"
    final_code += f"    {response}"
    final_code += f"return fig"
    final_code += f"\n\nfig = fig()"

    if submit_button:
        st.session_state.undo_code.append(final_code)
        # st.write(list(st.session_state.undo_code))
        # st.write(list(st.session_state.redo_code))
        exec(st.session_state.undo_code[-1])

        draw_plotly_chart(fig)
        show_executed_code()

elif prompt_input and st.session_state.undo_code and MODEL == "code-cushman-001":
    prompt = f"# Old fig()\n{st.session_state.undo_code[-1]}"
    prompt += "\n\n"
    prompt += "# " + prompt_input + "\n"
    prompt += "# New fig()\n"
    prompt += "def fig():\n"
    # st.text("New prompt:\n" + prompt)
    output = get_completion_response(
        model=MODEL,
        prompt=prompt,
        max_tokens=512,
        temperature=0,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        echo=False,
        stop=["return fig"],
    )
    response = output.choices[0].text
    final_code = "def fig():\n"
    final_code += f"{response}"
    final_code += f"return fig"
    final_code += f"\n\nfig = fig()"

    if submit_button:
        st.session_state.undo_code.append(final_code)
    # st.write(list(st.session_state.undo_code))
    # st.write(list(st.session_state.redo_code))
    exec(st.session_state.undo_code[-1])

    draw_plotly_chart(fig)
    show_executed_code()
