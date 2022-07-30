import pandas as pd
from plotly import subplots
from plotly.graph_objs import Bar, Histogram


def create_genre_distribution_plot(df: pd.DataFrame):
    """
    Function that creates a plotly graph of the genre distribution.

    @param df: The DataFrame containing the data.
    @return: The plotly graph.
    """

    genre_counts = df["genre"].value_counts()
    genre_names = genre_counts.index
    genre_counts = genre_counts.values

    return {
        "data": [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],

        "layout": {
            "title": "Distribution of Message Genres",
            "yaxis": {
                "title": "Count"
            },
            "xaxis": {
                "title": "Genre"
            },
            "height": 600,
            "font": {
                "size": 18
            }
        }
    }


def create_message_length_plot(df: pd.DataFrame):
    """
    Function that creates a plotly graph of the message length distribution.

    @param df: The DataFrame containing the data.
    @return: The plotly graph.
    """

    message_length_data = df["message"].str.len()
    message_length_data = message_length_data[message_length_data < 1000]

    return {
        "data": [
            Histogram(
                y=message_length_data
            )
        ],

        "layout": {
            "title": "Histogram of Message Lengths",
            "yaxis": {
                "title": "Count"
            },
            "xaxis": {
                "title": "Message Length"
            },
            "height": 600,
            "font": {
                "size": 18
            }
        }
    }


def create_target_distribution_plot(df: pd.DataFrame):
    """
    Function that creates a plotly graph of the target distribution.

    @param df: The DataFrame containing the data.
    @return: The plotly graph.
    """

    target_distribution_1 = df.drop(columns=["id", "message", "original", "genre"]).mean()
    target_distribution_0 = 1 - target_distribution_1  # Inverse the values, therefore to start from 0 and end with 1.
    class_names = target_distribution_1.index

    return {
        "data": [
            Bar(
                x=class_names,
                y=target_distribution_1,
                name="Class = 1"
            ),
            Bar(
                x=class_names,
                y=target_distribution_0,
                name="Class = 0",
                marker=dict(
                    color="rgb(212, 228, 247)"
                )
            )
        ],
        "layout": {
            "title": "Distribution of Each Target Class",
            "yaxis": {
                "title": "Distribution"
            },
            "xaxis": {
                "title": "Target",
            },
            "barmode": "stack",
            "height": 600,
            "font": {
                "size": 18
            },
            "margin": {
                "b": 300
            }
        }
    }


def create_genre_target_distribution_plot(df: pd.DataFrame):
    """
    Function that creates a plotly graph of the target distribution relative to the genre distribution.

    @param df: The DataFrame containing the data.
    @return: The plotly graph.
    """

    target_distribution_1 = df.drop(columns=["id", "message", "original"]).groupby("genre").mean().transpose()
    target_distribution_0 = 1 - target_distribution_1  # Inverse the values, therefore to start from 0 and end with 1.

    class_names = target_distribution_1.index
    genre_names = target_distribution_1.columns

    fig = subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        subplot_titles=genre_names
    )
    for i, genre in enumerate(genre_names):
        fig.add_trace(
            Bar(
                x=class_names,
                y=target_distribution_1[genre],
                name="Class = 1",
            ),
            row=i + 1, col=1
        )
        fig.add_trace(
            Bar(
                x=class_names,
                y=target_distribution_0[genre],
                name="Class = 0",
                marker=dict(
                    color="rgb(212, 228, 247)"
                )
            ),
            row=i + 1, col=1
        )
    fig.update_layout(**{
        "title": "Distribution of Each Target Class Relative to Genre",
        "barmode": "stack",
        "height": 600,
        "showlegend": False,
        "font": {
            "size": 18
        }
    })
    fig.update_xaxes(title_text="Target", row=3, col=1)
    fig.update_yaxes(title_text="Distribution", row=2, col=1)

    return fig
