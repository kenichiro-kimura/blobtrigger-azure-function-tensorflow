#  blobtrigger-azure-function-tensorflow

## Abstract

This is an Azure Functions project predicting whether a door image exprted Soracame is open or close.
If the door is open, send a image to the LINE Notify.

## How to use

1. Create an Azure Function App and Storage Account
2. Deploy the project to the Function App using GitHub Actions
3. Get a connection string of the Storage Account
4. Add a connection stiring to Function App settings as `StorageConnectionString`
5. Create a [LINE Notify](https://notify-bot.line.me/ja/) token
6. Add a token to  Function App settings as 'LineToken'
