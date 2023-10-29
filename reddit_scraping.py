import praw
import csv

post_num = 1000

# Reddit API credentials
CLIENT_ID = "H-4dRHrbjYWOUf0Pt4HntA"
CLIENT_SECRET = "fO9tlv7PAIsgVGw_K4XB58b402mKzw"
USER_AGENT = True  # e.g., "my_app/0.0.1"

# Initialize Reddit API client
# reddit = praw.Reddit(
#     client_id=CLIENT_ID,
#     client_secret=CLIENT_SECRET,
#     user_agent=USER_AGENT
# )

reddit = praw.Reddit(user_agent=True, client_id="H-4dRHrbjYWOUf0Pt4HntA",
  client_secret="fO9tlv7PAIsgVGw_K4XB58b402mKzw", username='CJ-Darty', password='Darty4Life')


# Get hot posts
hot_posts = reddit.subreddit("all").hot(limit=post_num)
data = []


for post in hot_posts:
    print("\n\n===========a new post!=========")
    print(post.title)
    print(post.num_comments)

    if post.num_comments > 0:
        post.comment_sort = 'best'  # Sorting comments by 'best'
        # post.comment_limit = 1  # Limiting to 1 comment

        print("###################")
        print(post.title)
        print(post.comments[0].body)
        data.append([post.title, post.comments[0].body])


# Write to CSV
with open(f"dataset/reddit/reddit_data_{post_num}.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["question", "answer"])
    writer.writerows(data)
