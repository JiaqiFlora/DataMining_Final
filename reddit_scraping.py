import praw
import csv

post_num = 1000

# Reddit API credentials
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
USER_AGENT = "YOUR_USER_AGENT"  # e.g., "my_app/0.0.1"

# Initialize Reddit API client
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    user_agent=USER_AGENT
)

# Get hot posts
hot_posts = reddit.front.hot(limit=post_num)  # Adjust limit as necessary

data = []

for post in hot_posts:
    if post.num_comments > 0:
        post.comment_sort = 'best'  # Sorting comments by 'best'
        post.comment_limit = 1  # Limiting to 1 comment

        print(post.comments)
        print(len(post.comments))

        for top_comment in post.comments:
            data.append([post.title, top_comment.body])

# Write to CSV
with open(f"dataset/reddit/reddit_data_{post_num}.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question", "answer"])
    writer.writerows(data)
