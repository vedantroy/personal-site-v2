import { unified } from "unified";
import remarkRehype from "remark-rehype";
import remarkParse from "remark-parse";
import rehypeKatex from "rehype-katex";
import rehypeStringify from "rehype-stringify/lib";
import remarkMath from "remark-math";

export default async function markdownToHtml(markdown: string) {
  const result = await unified()
    .use(remarkParse)
    .use(remarkMath)
    .use(remarkRehype)
    .use(rehypeStringify)
    .use(rehypeKatex)
    .use(rehypeStringify)
    .process(markdown);
  return result.toString();
}
