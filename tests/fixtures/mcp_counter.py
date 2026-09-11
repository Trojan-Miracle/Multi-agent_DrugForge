from mcp.server.fastmcp import FastMCP
mcp = FastMCP('test-counter')
counter = 0
@mcp.tool()
def count() -> int:
    global counter
    counter += 1
    return counter
if __name__ == '__main__':
    mcp.run(transport='stdio')
