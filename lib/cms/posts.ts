import fs from "fs";
import { join } from "path";
import matter from "gray-matter";

export type Post = {
  title: string;
  content: string;
  date: Date;
};

const postsDirectory = join(process.cwd(), "_posts");

export function getPostSlugs() {
  return fs.readdirSync(postsDirectory).map((x) => x.replace(/\.md$/, ""));
}

export function getPostBySlug(slug: string): Post {
  const fullPath = join(postsDirectory, `${slug}.md`);
  const fileContents = fs.readFileSync(fullPath, "utf8");
  const {
    data: { date, title },
    content,
  } = matter(fileContents);

  return { content, date, title };
}

export function getAllPosts() {
  return (
    getPostSlugs()
      .map((slug) => getPostBySlug(slug))
      // sort posts by date in descending order
      .sort((post1, post2) => (post1.date > post2.date ? -1 : 1))
  );
}
