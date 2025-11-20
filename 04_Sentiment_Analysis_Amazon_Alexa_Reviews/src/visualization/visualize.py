import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

df = pd.read_csv("../../data/raw/amazon_alexa.tsv", delimiter="\t", quoting=3)

# ------------------------
# Rating Column
# ------------------------

df["rating"].value_counts().plot(kind="bar", color="orange")
plt.title("Rating distribution count")
plt.xlabel("Ratings")
plt.ylabel("Count")
plt.show()

rating_counts = df["rating"].value_counts()
plt.pie(
    rating_counts,
    labels=rating_counts.index,
    shadow=True,
    autopct="%1.1f%%",
    startangle=90,
    colors=["skyblue", "orange", "green", "red", "purple"],
    explode=[0.1, 0.1, 0.1, 0.1, 0.1],
)
plt.title("Rating Distribution")
plt.savefig("../../reports/figures/rating_distribution.png")
plt.show()

# -----------------------
# Feedback Column
# -----------------------

# Check wether the rating is imbalanced
feedback_counts = df["feedback"].value_counts()
plt.pie(
    feedback_counts,
    labels=feedback_counts.index,
    explode=[0.1, 0.1],
    shadow=True,
    autopct="%1.1f%%",
    startangle=90,
    colors=["skyblue", "orange"],
)
plt.title("Feedback Distribution")
plt.savefig("../../reports/figures/feedback_distribution.png")
plt.show()

df["feedback"].value_counts().plot(kind="bar", color="blue")
plt.title("Feedback Distrubution Count")
plt.xlabel("Feedback")
plt.ylabel("Count")
plt.show()

# ------------------------
# Variation Column
# ------------------------

df["variation"].value_counts().plot(kind="barh", color="green")
plt.title("Variation Distribution")
plt.savefig("../../reports/figures/variation_distribution.png")
plt.show()

df.groupby("variation")["rating"].mean().plot(kind="barh", color="green")
plt.title("Mean Rating According to Variation")
plt.show()

# ------------------------
# Verified Reviews Column
# ------------------------

# Check the length of the 'verified_reviews' column
df["verified_reviews"].str.len().describe()
df["verified_reviews"].str.len().hist(bins=50, color="red")

df.dropna(inplace=True)
df["length"] = df["verified_reviews"].apply(len)

fig, ax = plt.subplots(figsize=(12, 6))
# Positive feedback
ax.hist(
    df[df["feedback"] == 1]["length"],
    bins=50,
    color="blue",
    label="Positive",
    alpha=0.5,
)
# Negative feedback
ax.hist(
    df[df["feedback"] == 0]["length"], bins=50, color="red", label="Negative", alpha=0.5
)
ax.set_title("Distribution of Review Lengths")
ax.set_xlabel("Review Length")
ax.set_ylabel("Frequency")
plt.show()

df.groupby("length")["rating"].mean().plot(kind="hist", bins=50, color="red")
plt.title("Mean Rating According to Length")
plt.show()

# ------------------------
# Date Column
# ------------------------

plt.figure(figsize=(8, 4))
df["date"].value_counts().plot(kind="line", color="purple")
plt.title("Date Distribution")
plt.xlabel("Date")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------
# Visualize Reviews
# ------------------------

# Create word cloud
wc = WordCloud(background_color="white", max_words=50)

# all reviews together
all_reviews = " ".join([review for review in df["verified_reviews"]])

# Visualize word cloud for all data
plt.figure(figsize=(7, 7))
plt.imshow(wc.generate(all_reviews))
plt.axis("off")
plt.title("Word Cloud for all Reviews")
plt.savefig("../../reports/figures/word_cloud_all_reviews.png")
plt.show()

# Positive reviews
positive_reviews = " ".join(
    [review for review in df[df["feedback"] == 1]["verified_reviews"]]
)

plt.figure(figsize=(7, 7))
plt.imshow(wc.generate(positive_reviews))
plt.axis("off")
plt.title("Word Cloud for Positive Reviews")
plt.savefig("../../reports/figures/word_cloud_positive_reviews.png")
plt.show()

# Negative reviews
negative_reviews = " ".join(
    [review for review in df[df["feedback"] == 0]["verified_reviews"]]
)

plt.figure(figsize=(7, 7))
plt.imshow(wc.generate(negative_reviews))
plt.axis("off")
plt.title("Word Cloud for Negative Reviews")
plt.savefig("../../reports/figures/word_cloud_negative_reviews.png")
plt.show()
