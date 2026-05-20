//+------------------------------------------------------------------+
//|                              AgroCast_QuantAgent_EA.mq5          |
//|                              AgroCast PRO - QuantAgent Lite      |
//|                              Automated Soybean Trading (Demo)    |
//+------------------------------------------------------------------+
#property copyright "AgroCast PRO"
#property link      "https://github.com/Nicotina712/agrocast"
#property version   "1.00"
#property description "QuantAgent-lite EA: reads LLM-generated signals from AgroCast API and executes trades on Sbean CFD."
#property strict

#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>

//--- Input parameters
input string   InpServerURL     = "https://TU-APP.onrender.com";  // AgroCast server URL (sin trailing slash)
input string   InpSymbol        = "Sbean_N6";     // Symbol (adjust per broker)
input double   InpLotSize       = 0.01;            // Lot size (min for demo)
input double   InpMaxRiskPct    = 2.0;             // Max risk % per trade
input int      InpMaxTradesDay  = 3;               // Max trades per day
input int      InpPollMinutes   = 30;              // Poll server every N minutes
input int      InpMaxSlippage   = 5;               // Max slippage points
input bool     InpAutoExecute   = true;            // Auto-execute signals (false = alert only)
input string   InpMagicComment  = "QA_LITE";       // Order comment for identification
input int      InpMagicNumber   = 20260520;        // Magic number

//--- Global variables
CTrade         trade;
CPositionInfo  posInfo;
datetime       lastPollTime = 0;
datetime       lastSignalTime = 0;
string         lastSignalDir = "FLAT";
int            tradesToday = 0;
datetime       currentDay = 0;
string         lastError = "";

//+------------------------------------------------------------------+
//| Expert initialization                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   // Validate inputs
   if(StringLen(InpServerURL) < 10)
   {
      Alert("ERROR: Configure la URL del servidor AgroCast!");
      return INIT_PARAMETERS_INCORRECT;
   }

   // Setup trade object
   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpMaxSlippage);
   trade.SetTypeFilling(ORDER_FILLING_IOC);

   // Allow WebRequest to our server
   Print("=== AgroCast QuantAgent EA v1.0 ===");
   Print("Server: ", InpServerURL);
   Print("Symbol: ", InpSymbol);
   Print("Lot: ", InpLotSize);
   Print("Auto-execute: ", InpAutoExecute ? "YES" : "NO (alert only)");
   Print("Poll every: ", InpPollMinutes, " min");
   Print("IMPORTANT: Add server URL to Tools -> Options -> Expert Advisors -> Allow WebRequest");
   Print("====================================");

   // Draw panel
   DrawInfoPanel("Initializing...", "FLAT", "", clrGray);

   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                           |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   ObjectsDeleteAll(0, "QA_");
   Comment("");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Reset daily counter
   MqlDateTime dt;
   TimeCurrent(dt);
   datetime today = StringToTime(IntegerToString(dt.year) + "." +
                                  IntegerToString(dt.mon) + "." +
                                  IntegerToString(dt.day));
   if(today != currentDay)
   {
      currentDay = today;
      tradesToday = 0;
   }

   // Poll server at configured interval
   datetime now = TimeCurrent();
   if(now - lastPollTime >= InpPollMinutes * 60)
   {
      lastPollTime = now;
      PollSignal();
   }

   // Update panel
   UpdatePanel();
}

//+------------------------------------------------------------------+
//| Poll the AgroCast API for latest signal                          |
//+------------------------------------------------------------------+
void PollSignal()
{
   string url = InpServerURL + "/api/quantagent/mt5signal";
   string headers = "Content-Type: application/json\r\n";
   char   postData[];
   char   result[];
   string resultHeaders;

   Print("[QA] Polling: ", url);

   int timeout = 10000; // 10 seconds
   int res = WebRequest("GET", url, headers, timeout, postData, result, resultHeaders);

   if(res == -1)
   {
      int err = GetLastError();
      lastError = "WebRequest failed: " + IntegerToString(err);
      if(err == 4014)
         lastError += " - Add URL to allowed list in Tools > Options > Expert Advisors";
      Print("[QA] ERROR: ", lastError);
      DrawInfoPanel(lastError, "ERROR", "", clrRed);
      return;
   }

   if(res != 200)
   {
      lastError = "HTTP " + IntegerToString(res);
      Print("[QA] HTTP error: ", res);
      DrawInfoPanel(lastError, "ERROR", "", clrRed);
      return;
   }

   // Parse JSON response
   string jsonStr = CharArrayToString(result);
   Print("[QA] Response: ", StringSubstr(jsonStr, 0, 200));

   // Extract signal fields from JSON
   string signal     = ExtractJsonString(jsonStr, "signal");
   string confidence = ExtractJsonString(jsonStr, "confidence");
   string timestamp  = ExtractJsonString(jsonStr, "timestamp");
   string volatility = ExtractJsonString(jsonStr, "volatility_regime");
   double entry      = ExtractJsonDouble(jsonStr, "entry");
   double sl         = ExtractJsonDouble(jsonStr, "stop_loss");
   double tp         = ExtractJsonDouble(jsonStr, "take_profit");
   int    contracts   = (int)ExtractJsonDouble(jsonStr, "contracts");

   lastError = "";

   // Log
   Print("[QA] Signal=", signal, " Conf=", confidence, " Entry=", entry,
         " SL=", sl, " TP=", tp, " Contracts=", contracts, " TS=", timestamp);

   // Update panel
   string info = "Signal: " + signal + " | Conf: " + confidence +
                 "\nEntry: " + DoubleToString(entry, 2) +
                 " | SL: " + DoubleToString(sl, 2) +
                 " | TP: " + DoubleToString(tp, 2) +
                 "\nVol: " + volatility + " | Time: " + timestamp;

   color panelColor = clrGray;
   if(signal == "LONG") panelColor = clrForestGreen;
   else if(signal == "SHORT") panelColor = clrFireBrick;

   DrawInfoPanel(info, signal, confidence, panelColor);

   // Check if this is a new signal
   if(timestamp == "" || timestamp == lastSignalTime)
   {
      Print("[QA] Same signal as before, skipping");
      return;
   }

   // Process signal
   if(signal == "FLAT" || signal == "")
   {
      lastSignalDir = "FLAT";
      Print("[QA] FLAT - no action");
      return;
   }

   // Avoid duplicate direction
   if(signal == lastSignalDir && HasOpenPosition())
   {
      Print("[QA] Already have ", signal, " position open");
      return;
   }

   // Daily limit
   if(tradesToday >= InpMaxTradesDay)
   {
      Print("[QA] Max trades/day reached (", InpMaxTradesDay, ")");
      return;
   }

   // Confidence filter
   if(confidence == "LOW")
   {
      Print("[QA] LOW confidence - skipping");
      return;
   }

   // Execute or alert
   if(InpAutoExecute)
   {
      ExecuteSignal(signal, entry, sl, tp, confidence);
   }
   else
   {
      string alertMsg = "QuantAgent Signal: " + signal +
                        " | Entry: " + DoubleToString(entry, 2) +
                        " | SL: " + DoubleToString(sl, 2) +
                        " | TP: " + DoubleToString(tp, 2) +
                        " | Conf: " + confidence;
      Alert(alertMsg);
      SendNotification(alertMsg);
   }

   lastSignalDir = signal;
   lastSignalTime = (datetime)StringToTime(timestamp);
}

//+------------------------------------------------------------------+
//| Execute a trade signal                                           |
//+------------------------------------------------------------------+
void ExecuteSignal(string signal, double entry, double sl, double tp, string confidence)
{
   // Close any existing opposite position first
   if(HasOpenPosition())
   {
      string posDir = GetPositionDirection();
      if((signal == "LONG" && posDir == "SHORT") ||
         (signal == "SHORT" && posDir == "LONG"))
      {
         Print("[QA] Closing opposite position before new signal");
         CloseAllPositions();
      }
      else
      {
         Print("[QA] Already aligned with signal direction");
         return;
      }
   }

   // Calculate lot size
   double lotSize = InpLotSize;

   // Get current price
   double ask = SymbolInfoDouble(InpSymbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(InpSymbol, SYMBOL_BID);
   double point = SymbolInfoDouble(InpSymbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(InpSymbol, SYMBOL_DIGITS);

   if(ask == 0 || bid == 0)
   {
      Print("[QA] ERROR: Cannot get price for ", InpSymbol);
      return;
   }

   // Normalize SL/TP
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   // Validate SL/TP distances
   double minStops = SymbolInfoInteger(InpSymbol, SYMBOL_TRADE_STOPS_LEVEL) * point;

   bool success = false;
   string comment = InpMagicComment + "_" + confidence;

   if(signal == "LONG")
   {
      // Validate SL is below ask
      if(sl >= ask)
      {
         Print("[QA] WARNING: SL (", sl, ") >= Ask (", ask, "), adjusting");
         sl = NormalizeDouble(ask - minStops - 1.0, digits);
      }
      if(tp <= ask && tp > 0)
      {
         Print("[QA] WARNING: TP (", tp, ") <= Ask (", ask, "), adjusting");
         tp = NormalizeDouble(ask + minStops + 1.0, digits);
      }

      Print("[QA] BUY ", lotSize, " @ ", ask, " SL=", sl, " TP=", tp);
      success = trade.Buy(lotSize, InpSymbol, ask, sl, tp, comment);
   }
   else if(signal == "SHORT")
   {
      // Validate SL is above bid
      if(sl <= bid)
      {
         Print("[QA] WARNING: SL (", sl, ") <= Bid (", bid, "), adjusting");
         sl = NormalizeDouble(bid + minStops + 1.0, digits);
      }
      if(tp >= bid && tp > 0)
      {
         Print("[QA] WARNING: TP (", tp, ") >= Bid (", bid, "), adjusting");
         tp = NormalizeDouble(bid - minStops - 1.0, digits);
      }

      Print("[QA] SELL ", lotSize, " @ ", bid, " SL=", sl, " TP=", tp);
      success = trade.Sell(lotSize, InpSymbol, bid, sl, tp, comment);
   }

   if(success)
   {
      tradesToday++;
      Print("[QA] Order executed successfully! Ticket: ", trade.ResultOrder());
      string msg = "AgroCast QA: " + signal + " executed @ " +
                   DoubleToString(signal == "LONG" ? ask : bid, digits);
      SendNotification(msg);
   }
   else
   {
      Print("[QA] Order FAILED: ", trade.ResultRetcodeDescription());
      lastError = "Order failed: " + trade.ResultRetcodeDescription();
   }
}

//+------------------------------------------------------------------+
//| Check if we have an open position for our symbol/magic           |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == InpSymbol && posInfo.Magic() == InpMagicNumber)
            return true;
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Get direction of current open position                           |
//+------------------------------------------------------------------+
string GetPositionDirection()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == InpSymbol && posInfo.Magic() == InpMagicNumber)
         {
            if(posInfo.PositionType() == POSITION_TYPE_BUY) return "LONG";
            if(posInfo.PositionType() == POSITION_TYPE_SELL) return "SHORT";
         }
      }
   }
   return "NONE";
}

//+------------------------------------------------------------------+
//| Close all positions for our symbol/magic                         |
//+------------------------------------------------------------------+
void CloseAllPositions()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      if(posInfo.SelectByIndex(i))
      {
         if(posInfo.Symbol() == InpSymbol && posInfo.Magic() == InpMagicNumber)
         {
            trade.PositionClose(posInfo.Ticket());
            Print("[QA] Closed position ticket ", posInfo.Ticket());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Simple JSON string extraction (no external library needed)       |
//+------------------------------------------------------------------+
string ExtractJsonString(string json, string key)
{
   string search = "\"" + key + "\"";
   int pos = StringFind(json, search);
   if(pos == -1) return "";

   // Find the colon after key
   int colonPos = StringFind(json, ":", pos + StringLen(search));
   if(colonPos == -1) return "";

   // Skip whitespace
   int start = colonPos + 1;
   while(start < StringLen(json) && (StringGetCharacter(json, start) == ' ' ||
         StringGetCharacter(json, start) == '\t')) start++;

   // Check if value is a string (starts with quote) or null
   ushort ch = StringGetCharacter(json, start);
   if(ch == '"')
   {
      start++;
      int end = StringFind(json, "\"", start);
      if(end == -1) return "";
      return StringSubstr(json, start, end - start);
   }
   else if(ch == 'n') // null
   {
      return "";
   }
   else
   {
      // Number or other - find delimiter
      int end = start;
      while(end < StringLen(json))
      {
         ushort c = StringGetCharacter(json, end);
         if(c == ',' || c == '}' || c == ']' || c == ' ' || c == '\n') break;
         end++;
      }
      return StringSubstr(json, start, end - start);
   }
}

//+------------------------------------------------------------------+
//| Extract a double value from JSON                                 |
//+------------------------------------------------------------------+
double ExtractJsonDouble(string json, string key)
{
   string val = ExtractJsonString(json, key);
   if(val == "" || val == "null") return 0;
   return StringToDouble(val);
}

//+------------------------------------------------------------------+
//| Draw on-chart info panel                                         |
//+------------------------------------------------------------------+
void DrawInfoPanel(string info, string signal, string confidence, color bgColor)
{
   string prefix = "QA_";

   // Background rectangle
   if(ObjectFind(0, prefix + "bg") < 0)
   {
      ObjectCreate(0, prefix + "bg", OBJ_RECTANGLE_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "bg", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "bg", OBJPROP_XDISTANCE, 10);
      ObjectSetInteger(0, prefix + "bg", OBJPROP_YDISTANCE, 30);
      ObjectSetInteger(0, prefix + "bg", OBJPROP_XSIZE, 350);
      ObjectSetInteger(0, prefix + "bg", OBJPROP_YSIZE, 110);
      ObjectSetInteger(0, prefix + "bg", OBJPROP_BORDER_TYPE, BORDER_FLAT);
   }
   ObjectSetInteger(0, prefix + "bg", OBJPROP_BGCOLOR, bgColor);
   ObjectSetInteger(0, prefix + "bg", OBJPROP_COLOR, clrWhite);

   // Title
   if(ObjectFind(0, prefix + "title") < 0)
   {
      ObjectCreate(0, prefix + "title", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "title", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "title", OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, prefix + "title", OBJPROP_YDISTANCE, 35);
      ObjectSetString(0, prefix + "title", OBJPROP_FONT, "Arial Bold");
      ObjectSetInteger(0, prefix + "title", OBJPROP_FONTSIZE, 11);
      ObjectSetInteger(0, prefix + "title", OBJPROP_COLOR, clrWhite);
   }
   ObjectSetString(0, prefix + "title", OBJPROP_TEXT,
                   "AgroCast QuantAgent | " + signal + " " + confidence);

   // Info text
   if(ObjectFind(0, prefix + "info") < 0)
   {
      ObjectCreate(0, prefix + "info", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "info", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "info", OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, prefix + "info", OBJPROP_YDISTANCE, 55);
      ObjectSetString(0, prefix + "info", OBJPROP_FONT, "Courier New");
      ObjectSetInteger(0, prefix + "info", OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, prefix + "info", OBJPROP_COLOR, clrWhite);
   }
   // Split multiline
   string lines[];
   int nLines = StringSplit(info, '\n', lines);
   ObjectSetString(0, prefix + "info", OBJPROP_TEXT,
                   nLines > 0 ? lines[0] : info);

   // Second line
   if(ObjectFind(0, prefix + "info2") < 0)
   {
      ObjectCreate(0, prefix + "info2", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "info2", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "info2", OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, prefix + "info2", OBJPROP_YDISTANCE, 72);
      ObjectSetString(0, prefix + "info2", OBJPROP_FONT, "Courier New");
      ObjectSetInteger(0, prefix + "info2", OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, prefix + "info2", OBJPROP_COLOR, clrWhite);
   }
   ObjectSetString(0, prefix + "info2", OBJPROP_TEXT,
                   nLines > 1 ? lines[1] : "");

   // Third line
   if(ObjectFind(0, prefix + "info3") < 0)
   {
      ObjectCreate(0, prefix + "info3", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "info3", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "info3", OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, prefix + "info3", OBJPROP_YDISTANCE, 89);
      ObjectSetString(0, prefix + "info3", OBJPROP_FONT, "Courier New");
      ObjectSetInteger(0, prefix + "info3", OBJPROP_FONTSIZE, 9);
      ObjectSetInteger(0, prefix + "info3", OBJPROP_COLOR, clrWhite);
   }
   ObjectSetString(0, prefix + "info3", OBJPROP_TEXT,
                   nLines > 2 ? lines[2] : "");

   // Status bar
   if(ObjectFind(0, prefix + "status") < 0)
   {
      ObjectCreate(0, prefix + "status", OBJ_LABEL, 0, 0, 0);
      ObjectSetInteger(0, prefix + "status", OBJPROP_CORNER, CORNER_LEFT_UPPER);
      ObjectSetInteger(0, prefix + "status", OBJPROP_XDISTANCE, 20);
      ObjectSetInteger(0, prefix + "status", OBJPROP_YDISTANCE, 115);
      ObjectSetString(0, prefix + "status", OBJPROP_FONT, "Arial");
      ObjectSetInteger(0, prefix + "status", OBJPROP_FONTSIZE, 8);
      ObjectSetInteger(0, prefix + "status", OBJPROP_COLOR, clrDarkGray);
   }
   ObjectSetString(0, prefix + "status", OBJPROP_TEXT,
                   "Trades today: " + IntegerToString(tradesToday) + "/" +
                   IntegerToString(InpMaxTradesDay) +
                   " | Last poll: " + TimeToString(lastPollTime, TIME_MINUTES) +
                   (lastError != "" ? " | ERR: " + lastError : ""));

   ChartRedraw();
}

//+------------------------------------------------------------------+
//| Update panel with current status                                 |
//+------------------------------------------------------------------+
void UpdatePanel()
{
   // Update status bar every tick
   if(ObjectFind(0, "QA_status") >= 0)
   {
      int secToNext = InpPollMinutes * 60 - (int)(TimeCurrent() - lastPollTime);
      if(secToNext < 0) secToNext = 0;

      string posStatus = "No position";
      if(HasOpenPosition())
      {
         for(int i = PositionsTotal() - 1; i >= 0; i--)
         {
            if(posInfo.SelectByIndex(i) && posInfo.Symbol() == InpSymbol &&
               posInfo.Magic() == InpMagicNumber)
            {
               posStatus = (posInfo.PositionType() == POSITION_TYPE_BUY ? "LONG" : "SHORT") +
                           " P&L: " + DoubleToString(posInfo.Profit(), 2);
               break;
            }
         }
      }

      ObjectSetString(0, "QA_status", OBJPROP_TEXT,
                      "Trades: " + IntegerToString(tradesToday) + "/" +
                      IntegerToString(InpMaxTradesDay) +
                      " | " + posStatus +
                      " | Next poll: " + IntegerToString(secToNext) + "s" +
                      (lastError != "" ? " | " + lastError : ""));
      ChartRedraw();
   }
}
//+------------------------------------------------------------------+
