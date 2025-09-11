import sys, logging
from mcp.server.fastmcp import FastMCP     # real SDK
from app_tools.web_search import image_search  # your tool function

from pathlib import Path

from app_tools.db_search import (
search_similar,
embed_folder,
create_or_update_index,
upload_docs,
DEFAULT_IMAGES_DIR,
DEFAULT_OUTPUT_JSON,
INDEX_NAME,
)

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

mcp = FastMCP("image_server")
mcp.tool(name="image_search",
         description="Return N related images for a query via SerpAPI")(image_search)



# @mcp.tool(name="index_images", description="Embed a folder of images and upload to Azure AI Search")
# async def index_images(
#     images_dir: str | None = None,
#     output_json: str | None = None,
#     recreate_index: bool = True
# ) -> dict:
#     """
#     1) Embeds images in 'images_dir' and writes doc vectors to 'output_json'
#     2) (Re)creates the Azure Search index (if recreate_index=True)
#     3) Uploads all docs
#     Returns counts and file paths. No stdout printing—use returned JSON.
#     """
#     dir_path = Path(images_dir) if images_dir else DEFAULT_IMAGES_DIR
#     out_path = Path(output_json) if output_json else DEFAULT_OUTPUT_JSON

#     result_embed = embed_folder(dir_path, out_path)

#     if recreate_index:
#         _ = create_or_update_index(INDEX_NAME)

#     result_upload = upload_docs(out_path, INDEX_NAME)

#     return {
#         "embedded_count": result_embed["count"],
#         "json_written_to": result_embed["docs_path"],
#         "uploaded": result_upload["uploaded"],
#         "index": INDEX_NAME
#     }

@mcp.tool(name="search_by_image", description="Return top-K similar images from Azure AI Search")
async def search_by_image(
    query_image_path: str,
    k: int = 3
) -> list[dict]:
    """
    Given a query image path (local or URL), returns top-K similar images
    with decoded local paths and existence flags.
    """
    return search_similar(query_image_path, k=k)

@mcp.tool()
async def health() -> str:
    return "ok"

if __name__ == "__main__":
    mcp.run(transport="stdio")

