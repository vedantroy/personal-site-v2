import clsx from "clsx";
import Head from "next/head";

interface LinkProps {
  children?: React.ReactNode;
  href: string;
  className?: string;
  italic?: boolean;
  underline?: boolean;
  bold?: boolean;
}

const Link = (
  { children, href, italic, bold, underline, className }: LinkProps = {
    italic: false,
    bold: false,
    underline: false,
    className: "",
    href: "",
  }
) => (
  <a
    href={href}
    target="_blank"
    rel="noopener noreferrer"
    className={clsx(
      italic && "italic",
      bold && "font-bold",
      underline && "underline",
      "hover:underline",
      className
    )}
  >
    {children}
  </a>
);
export default function Home() {
  return (
    <>
      <Head>
        <title>Vedant Roy</title>
      </Head>
      <div className="m-32">
        <div className="mb-2">
          <Link href="https://twitter.com/vroomerify?lang=en" bold>
            Vedant Roy
          </Link>
        </div>
        <div className="mb-2">
          <div className="text-gray-700 mr-2 mb-1">Projects</div>
          <div className="flex flex-col ml-4 gap-2">
            <div>
              <span className="">Contrastive loss for Pokemon </span>{" "}
              <Link href="/pokemon.mp4" bold>
                video search
              </Link>
            </div>
            <div>
              Diffusion model for{" "}
              <Link
                href="https://github.com/vedantroy/improved-ddpm-pytorch"
                bold
              >
                face generation
              </Link>
            </div>
            <div>
              Websites for{" "}
              <Link
                href="https://twitter.com/vroomerify/status/1521806346881929216"
                bold
              >
                collaborative
              </Link>{" "}
              <Link href="https://webhighlighter.com" bold>
                learning
              </Link>
            </div>
            <div>
              Automatic runtime type-checking with{" "}
              <Link href="https://github.com/vedantroy/typecheck.macro" bold>
                a compiler
              </Link>
            </div>
            <div>
              Several{" "}
              <Link
                href="https://github.com/babel/babel/pulls?q=is%3Apr+author%3Avedantroy"
                bold
              >
                open-source PRs
              </Link>
            </div>
            <div>
              250K+ and 3K+ downloaded{" "}
              <span className="font-bold">mobile apps</span>
            </div>
          </div>
        </div>
        <div>
          <div className="text-gray-700 mr-2 mb-1">Posts</div>
          <div className="flex flex-col ml-4 gap-2">
            <div>
              <Link href="/posts/diffusion" italic>
                Diffusion model math for beginners (WIP)
              </Link>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
