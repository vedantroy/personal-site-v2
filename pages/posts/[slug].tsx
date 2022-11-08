import Head from "next/head";
import { getPostBySlug, getPostSlugs } from "../../lib/cms/posts";
import type { Post } from "../../lib/cms/posts";
import markdownToHtml from "../../lib/doc/markdown2html";

type Props = {
  post: Post;
};

export default function Post({ post }: Props) {
  return (
    <>
      <article className="py-4 max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
        <Head>
          <title>{post.title}</title>
          <link
            href="https://cdn.jsdelivr.net/npm/katex@0.15.0/dist/katex.min.css"
            rel="stylesheet"
          />
        </Head>
        <div id="container" className="flex flex-col items-center">
          <div
            className="prose"
            dangerouslySetInnerHTML={{ __html: post.content }}
          ></div>
        </div>
      </article>
    </>
  );
}

type Params = {
  params: {
    slug: string;
  };
};

export async function getStaticProps({ params }: Params) {
  const post = getPostBySlug(params.slug);
  post.content = await markdownToHtml(post.content);
  return {
    props: { post },
  };
}

export function getStaticPaths() {
  return {
    paths: getPostSlugs().map((post) => {
      return {
        params: {
          slug: post,
        },
      };
    }),
    fallback: false,
  };
}
