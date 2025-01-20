import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_genres_popularity(genre_trends_grouped, top_n=10):
    """
    Plots the popularity of genres over time, based on total votes, for the top N genres.

    Parameters:
    ----------
    genre_trends_grouped (DataFrame): Grouped data containing genre trends.
    top_n (int): The number of top genres to display.
    """
    
    genre_trends_pd = genre_trends_grouped.toPandas()
    genre_trends_pd = genre_trends_pd.sort_values(by=["genre", "startYear"])
    genres = genre_trends_pd.groupby("genre")["totalVotes"].sum().nlargest(top_n).index

    fig = go.Figure()

    for genre in genres:
        genre_data = genre_trends_pd[genre_trends_pd["genre"] == genre]

        fig.add_trace(go.Scatter(
            x=genre_data["startYear"],
            y=genre_data["totalVotes"],
            mode="lines+markers",
            name=genre,
            hovertemplate=(
                f"<b>Genre:</b> {genre}<br>"
                "<b>Year:</b> %{x}<br>"
                "<b>Total Votes:</b> %{y}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Popularity by Genre Over Time (Top 10 by Votes Count)",
        xaxis_title="Year",
        yaxis_title="Total Votes",
        legend_title="Genre",
        hovermode="x unified",
        height=700
    )

    fig.show()


def plot_genres_rating(genre_trends_grouped, top_n=10):
    """
    Plots the average rating of genres over time, for the top N genres.

    Parameters:
    ----------
    genre_trends_grouped (DataFrame): Grouped data containing genre trends.
    top_n (int): The number of top genres to display.
    """

    genre_trends_pd = genre_trends_grouped.toPandas()
    genre_trends_pd = genre_trends_pd.sort_values(by=["genre", "startYear"])
    genres = genre_trends_pd.groupby("genre")["totalVotes"].sum().nlargest(top_n).index

    fig = go.Figure()

    for genre in genres:
        genre_data = genre_trends_pd[genre_trends_pd["genre"] == genre]

        fig.add_trace(go.Scatter(
            x=genre_data["startYear"],
            y=genre_data["avgRating"],
            mode="lines+markers",
            name=genre,
            hovertemplate=(
                f"<b>Genre:</b> {genre}<br>"
                "<b>Year:</b> %{x}<br>"
                "<b>Average Rating:</b> %{y}<extra></extra>"
            )
        ))

    fig.update_layout(
        title="Average Rating by Genre Over Time (Top 10 by Votes Count)",
        xaxis_title="Year",
        yaxis_title="Average Rating",
        legend_title="Genre",
        hovermode="x unified",
        height=700
    )

    fig.show()


def plot_genres_interactive(genre_trends_pd):
    """
    Creates an interactive plot that shows both total votes and average ratings for genres over time.

    Parameters:
    ----------
    genre_trends_pd (DataFrame): Data containing genre trends, including total votes and average ratings.
    """

    # Sort and filter the data
    df = genre_trends_pd.sort_values(by=["genre", "startYear"])

    # Create a subplot layout with two horizontally arranged subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Total Votes Over Time", "Average Rating Over Time"),
        shared_xaxes=True,
        vertical_spacing=0.15
    )

    genres = df["genre"].unique()

    # Add traces for totalVotes to the first subplot
    for genre in genres:
        genre_data = df[df["genre"] == genre]
        fig.add_trace(
            go.Scatter(
                x=genre_data["startYear"],
                y=genre_data["totalVotes"],
                mode="lines+markers",
                name=genre,
                visible=False  # Hide all traces initially
            ),
            row=1, col=1
        )

    # Add traces for avgRating to the second subplot
    for genre in genres:
        genre_data = df[df["genre"] == genre]
        fig.add_trace(
            go.Scatter(
                x=genre_data["startYear"],
                y=genre_data["avgRating"],
                mode="lines+markers",
                name=genre,
                visible=False  # Hide all traces initially
            ),
            row=2, col=1
        )

    # Create a button for each genre to control visibility of traces
    buttons = []
    for i, genre in enumerate(genres):
        visibility = [False] * len(fig.data)
        # Set visibility for the traces of the current genre
        visibility[i] = True  # For totalVotes
        visibility[i + len(genres)] = True  # For avgRating
        buttons.append(
            dict(
                label=genre,
                method="update",
                args=[{"visible": visibility}]
            )
        )

    # Set the first traces to visible by default
    for i in range(len(genres)):
        fig.data[i].visible = (i == 0)  # First trace for totalVotes
        fig.data[i + len(genres)].visible = (i == 0)  # First trace for avgRating

    # Update layout with dropdown menu and titles
    fig.update_layout(
        updatemenus=[{
            "buttons": buttons,
            "direction": "down",
            "showactive": True,
            "x": 1,
            "y": 1.1
        }],
        title="Popularity and Ratings by Genre Over Time",
        xaxis_title="Year",
        yaxis=dict(title="Total Votes"),
        xaxis2_title="Year",
        yaxis2=dict(title="Average Rating"),
        showlegend=False,
        height=700,
        width=1100
    )

    fig.show()


