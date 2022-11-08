import clsx from "clsx";

const Link = ({ children, url, italic, bold }) => (
  <a
    href={url}
    className={clsx(italic && "italic", bold && "bold", "hover:underline")}
  >
    {children}
  </a>
);
export default function Home() {
  return (
    <div className="m-32">
      <div className="flex flex-row">
        <div className="text-gray-700 mr-2">Name:</div>
        <span className="font-bold">Vedant Roy</span>
      </div>
      <div className="flex flex-row">
        <div className="text-gray-700 mr-2">Links:&nbsp;</div>
        <div className="flex flex-row">
          <div className="flex flex-row mr-1">
            <Link italic url="hi">
              Github
            </Link>
            <span>,</span>
          </div>
          <Link italic url="hi">
            Twitter
          </Link>
        </div>
      </div>
      <div>
        <div className="text-gray-700 mr-2 mb-2">Projects:</div>
        <div className="flex flex-col ml-4 gap-2">
          <div>
            Pokemon <Link>video search</Link> using contrastive loss
          </div>
          <div>
            <Link>Face generation</Link> with diffusion models
          </div>
          <div>Collaborative learning (Chimu, Web Highlighter)</div>
          <div>
            A compiler for generating Typescript type validation functions
          </div>
          <div>A bunch of open-source PRs</div>
          <div>
            (High-school) Assorted mobile apps (250K+ downloads, 3K+, 1K+)
          </div>
        </div>
      </div>
      <div className="flex flex-row">
        <div className="text-gray-700 mr-2">Posts:</div>
      </div>
    </div>
  );
}
